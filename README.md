# Face Recognition from Partial Face

Experiment code behind the presentation. Pipeline: LFW → MTCNN detection + 5-point alignment →
partial-face variants → frozen VGG-Face / FaceNet features → Softmax classifier.

## Layout

```
src/common.py                 settings, alignment template, occlusion variants
src/vggface.py                VGG-Face (VGG-16) in PyTorch, loads keras-vggface weights
src/step1_detect_align.py     pick identities, MTCNN detect + align -> results/faces.npz
src/step2_extract_features.py build 7 variants, extract features -> results/features_*.npz
src/step3_train_evaluate.py   train M0-M4, evaluate per variant -> results/metrics.json
src/step4_figures.py          figures for the slides -> results/figures/*.png
data/      LFW (downloaded by scikit-learn's fetch_lfw_people into data/lfw_home)
weights/   rcmalli_vggface_tf_vgg16.h5 from github.com/rcmalli/keras-vggface releases
```

## Run

```bash
pip install --user torch torchvision scikit-learn opencv-python-headless matplotlib h5py
pip install --user --no-deps facenet-pytorch==2.6.0
cd src
python step1_detect_align.py
python step2_extract_features.py
python step3_train_evaluate.py
python step4_figures.py
```

To use your own photos (e.g. the 5 group members), put them in `data/lfw_home/lfw/<person>/*.jpg`
(or point `LFW_DIR` in `step1_detect_align.py` at a folder with one sub-folder per person) and
lower `MIN_IMGS` to match how many photos each person has.
