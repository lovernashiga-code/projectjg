"""Download the datasets from Kaggle (slide 6) and the pretrained VGG-Face weights from GitHub.

  pip install kagglehub
  python download_data.py                 # LFW + FEI + VGG-Face weights
  python download_data.py fei weights     # only some of them

  lfw      kaggle.com/datasets/jessicali9530/lfw-dataset  -> data/lfw/
  fei      kaggle.com/datasets/whizzkid/fei-face-data     -> data/fei/
  weights  github.com/rcmalli/keras-vggface (v2.0)        -> weights/rcmalli_vggface_tf_vgg16.h5

If Kaggle asks for a login, create an API token (kaggle.com > Settings > API > Create New Token)
and put kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<you>\\.kaggle\\), or set KAGGLE_USERNAME
and KAGGLE_KEY. FaceNet and MTCNN weights come with facenet-pytorch and need no download.
Note: an existing data/lfw_home/lfw (scikit-learn download used for the slides) takes priority
over data/lfw/ in step 1, so the slide numbers stay reproducible.
"""
import argparse
import shutil
import urllib.request

from common import DATA, WEIGHTS

KAGGLE = {"lfw": "jessicali9530/lfw-dataset", "fei": "whizzkid/fei-face-data"}
VGGFACE = ("https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5",
           WEIGHTS / "rcmalli_vggface_tf_vgg16.h5")


def fetch_kaggle(name):
    dst = DATA / name
    if dst.exists() and any(dst.iterdir()):
        print(f"{dst} already exists, skipping")
        return
    import kagglehub
    src = kagglehub.dataset_download(KAGGLE[name])
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"{name}: {src} -> {dst}")


def fetch_weights():
    url, dst = VGGFACE
    if dst.exists():
        print(f"{dst} already exists, skipping")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")

    def progress(blocks, size, total):
        print(f"\rVGG-Face weights {blocks * size / 2**20:6.0f} / {total / 2**20:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, progress)
    tmp.replace(dst)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="*", help="lfw, fei, weights (default: all)")
    for w in ap.parse_args().what or ["lfw", "fei", "weights"]:
        fetch_weights() if w == "weights" else fetch_kaggle(w)
