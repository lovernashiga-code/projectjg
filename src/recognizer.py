"""Recognise faces in new images with a classifier saved by step 3 (models/<dataset>_<method>.joblib).

MTCNN -> 5-point alignment -> optional synthetic occlusion -> frozen backbone -> Softmax.
When the top probability is below tau the answer is "Unknown" (slide 9).
Used by predict.py, webcam.py and app.py.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from facenet_pytorch import MTCNN
from sklearn.preprocessing import normalize

from backbones import extract, load_backbone
from common import MODELS, UNKNOWN, align, apply_variant


@dataclass
class Face:
    box: np.ndarray        # x0, y0, x1, y1 in the input image
    landmarks: np.ndarray  # 5 x 2: eyes, nose, mouth corners
    face: np.ndarray       # the aligned 224x224 RGB face the classifier saw (after the occlusion)
    name: str              # predicted person, or UNKNOWN when prob < tau
    prob: float            # Softmax probability of the best class
    top: list              # [(name, prob), ...] best first


def default_model():
    """models/lfw_M2.joblib if present, else the first VGG-Face full+partial model, else any model."""
    for pattern in ("lfw_M2.joblib", "*_M2.joblib", "*.joblib"):
        found = sorted(MODELS.glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(f"no trained model in {MODELS}: run step1 to step3 first (see README)")


class Recognizer:
    def __init__(self, model_path=None, tau=None, min_face=40, min_detect=0.95):
        self.path = Path(model_path) if model_path else default_model()
        self.bundle = joblib.load(self.path)
        self.clf, self.names = self.bundle["classifier"], self.bundle["names"]
        self.tau = self.bundle["tau"] if tau is None else tau
        self.model, self.prep = load_backbone(self.bundle["backbone"])
        self.mtcnn = MTCNN(keep_all=True, min_face_size=min_face, device="cpu")
        self.min_detect = min_detect   # MTCNN face probability; drops background false detections

    def describe(self):
        b = self.bundle
        hint = (f" (tau {b['tau_strangers_5pct']} lets at most 5% of strangers through)"
                if b.get("tau_strangers_5pct") is not None else "")
        return (f"{self.path.name}: {b['method_name']}, {len(self.names)} people, "
                f"fit on {b['fit_on']}, tau={self.tau}{hint}")

    def __call__(self, rgb, variant="full", tau=None, top=5):
        """rgb: HxWx3 uint8 image. Returns one Face per detected face, largest first."""
        tau = self.tau if tau is None else tau
        rgb = np.ascontiguousarray(rgb[..., :3])
        boxes, det, lms = self.mtcnn.detect(rgb, landmarks=True)
        if boxes is None:
            return []
        keep = np.asarray(det, np.float32) >= self.min_detect
        boxes, lms = np.asarray(boxes, np.float32)[keep], np.asarray(lms, np.float32)[keep]
        if not len(boxes):
            return []
        order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        boxes, lms = boxes[order], lms[order]
        faces = np.stack([apply_variant(align(rgb, p), variant) for p in lms])
        prob = self.clf.predict_proba(normalize(extract(self.model, self.prep, faces)))
        out = []
        for box, pts, face, p in zip(boxes, lms, faces, prob):
            best = np.argsort(-p)[:top]
            ranked = [(self.names[self.clf.classes_[j]], float(p[j])) for j in best]
            name = ranked[0][0] if ranked[0][1] >= tau else UNKNOWN
            out.append(Face(box, pts, face, name, ranked[0][1], ranked))
        return out


def draw(bgr, faces):
    """Draw boxes and labels on a BGR image in place (green = recognised, red = unknown)."""
    for f in faces:
        x0, y0, x1, y1 = f.box.astype(int)
        color = (70, 170, 60) if f.name != UNKNOWN else (60, 60, 220)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
        label = f"{f.name.replace('_', ' ')} {f.prob:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(y0, h + 8)
        cv2.rectangle(bgr, (x0, y - h - 8), (x0 + w + 6, y), color, -1)
        cv2.putText(bgr, label, (x0 + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return bgr
