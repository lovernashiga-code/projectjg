"""VGG-Face (VGG-16, Parkhi et al. 2015) in PyTorch, loading the keras-vggface (rcmalli) weights."""
import h5py
import numpy as np
import torch
import torch.nn as nn

CONV = [(3, 64), (64, 64), "M", (64, 128), (128, 128), "M",
        (128, 256), (256, 256), (256, 256), "M",
        (256, 512), (512, 512), (512, 512), "M",
        (512, 512), (512, 512), (512, 512), "M"]
CONV_NAMES = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
              "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]
# keras-vggface "version 1" preprocessing: BGR channel means
MEAN_BGR = torch.tensor([93.5940, 104.7624, 129.1863]).view(1, 3, 1, 1)


class VGGFace(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for c in CONV:
            if c == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(c[0], c[1], 3, padding=1), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def forward(self, x):
        """x: uint8-range RGB tensor (N, 3, 224, 224). Returns the 4096-d fc7 descriptor."""
        x = x[:, [2, 1, 0]] - MEAN_BGR
        x = self.features(x).flatten(1)
        x = torch.relu(self.fc6(x))
        return torch.relu(self.fc7(x))


def _get(f, name):
    g = f[name]
    while isinstance(g, h5py.Group) and len(g) == 1 and isinstance(next(iter(g.values())), h5py.Group):
        g = next(iter(g.values()))
    keys = sorted(g.keys())
    w = next(k for k in keys if "kernel" in k or k.endswith("_W") or k.startswith("W"))
    b = next(k for k in keys if "bias" in k or k.endswith("_b") or k.startswith("b"))
    return np.array(g[w]), np.array(g[b])


def load_vggface(path):
    model = VGGFace()
    convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
    with h5py.File(path, "r") as f:
        root = f["model_weights"] if "model_weights" in f else f
        for m, name in zip(convs, CONV_NAMES):
            w, b = _get(root, name)                       # (3, 3, in, out)
            m.weight.data = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
            m.bias.data = torch.from_numpy(b)
        w, b = _get(root, "fc6")                          # (7*7*512, 4096), HWC flatten order
        w = w.reshape(7, 7, 512, 4096).transpose(2, 0, 1, 3).reshape(-1, 4096)
        model.fc6.weight.data = torch.from_numpy(w.T.copy())
        model.fc6.bias.data = torch.from_numpy(b)
        for layer, name in [(model.fc7, "fc7"), (model.fc8, "fc8")]:
            w, b = _get(root, name)
            layer.weight.data = torch.from_numpy(w.T.copy())
            layer.bias.data = torch.from_numpy(b)
    return model.eval()
