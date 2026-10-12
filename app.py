"""Streamlit demo: upload a photo or take one with the camera, add a synthetic occlusion, see who it is.

  streamlit run app.py

Needs a trained model in models/ (src/step3_train_evaluate.py) and, for VGG-Face models,
weights/rcmalli_vggface_tf_vgg16.h5.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import MODELS, UNKNOWN, VARIANT_TH, VARIANTS  # noqa: E402
from recognizer import Recognizer, draw  # noqa: E402

st.set_page_config(page_title="Partial Face Recognition", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def load(path):
    return Recognizer(path)


st.title("Face Recognition from Partial Face")
st.caption("MTCNN → Alignment → VGG-Face / FaceNet (frozen) → Softmax · ถ้าความมั่นใจต่ำกว่า τ ระบบตอบ \"ไม่รู้จัก\"")

models = sorted(MODELS.glob("*.joblib"))
if not models:
    st.error("ยังไม่มีโมเดลใน models/ — รัน src/step1 ถึง step3 ก่อน (ดู README)")
    st.stop()

with st.sidebar:
    preferred = next((i for i, p in enumerate(models) if p.name == "lfw_M2.joblib"),
                     next((i for i, p in enumerate(models) if p.stem.endswith("_M2")), 0))
    path = st.selectbox("Model", models, index=preferred, format_func=lambda p: p.stem)
    rec = load(str(path))
    st.caption(f"{rec.bundle['method_name']} · {len(rec.names)} คน · fit on {rec.bundle['fit_on']}")
    tau = st.slider("เกณฑ์ความมั่นใจ τ", 0.0, 0.99, float(rec.bundle["tau"]), 0.05)
    if rec.bundle.get("tau_strangers_5pct") is not None:
        st.caption(f"τ ≥ {rec.bundle['tau_strangers_5pct']:.2f}: คนแปลกหน้าถูกตอบเป็นชื่อไม่เกิน 5 % (step 3)")
    variant = st.radio("จำลองการบดบัง", list(VARIANTS), format_func=lambda v: f"{VARIANT_TH[v]} ({v})")
    source = st.radio("ภาพ", ["อัปโหลดรูป", "กล้อง"], horizontal=True)

file = (st.file_uploader("รูปที่มีใบหน้า", type=["jpg", "jpeg", "png", "bmp"]) if source == "อัปโหลดรูป"
        else st.camera_input("ถ่ายรูป"))
if file is None:
    st.info("อัปโหลดรูปหรือถ่ายรูปเพื่อเริ่ม")
    st.stop()

rgb = np.array(ImageOps.exif_transpose(Image.open(file)).convert("RGB"))
with st.spinner("Recognising ..."):
    faces = rec(rgb, variant, tau=tau)

left, right = st.columns([3, 2])
left.image(draw(rgb[..., ::-1].copy(), faces)[..., ::-1], caption=f"พบ {len(faces)} ใบหน้า", width="stretch")
if not faces:
    right.warning("MTCNN ไม่พบใบหน้าในรูปนี้")
for i, f in enumerate(faces, 1):
    with right.container(border=True):
        c1, c2 = st.columns([1, 2])
        c1.image(f.face, caption=f"ใบหน้า {i} ({variant})", width="stretch")
        name = "ไม่รู้จัก (Unknown)" if f.name == UNKNOWN else f.name.replace("_", " ")
        c2.markdown(f"### {name}")
        c2.markdown(f"ความมั่นใจ **{f.prob:.1%}** (τ = {tau:.2f})")
        c2.dataframe({"Top-5": [n.replace("_", " ") for n, _ in f.top], "p": [p for _, p in f.top]},
                     column_config={"p": st.column_config.ProgressColumn("p", min_value=0.0, max_value=1.0,
                                                                         format="%.2f")},
                     hide_index=True, width="stretch")
