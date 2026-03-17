

import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

AUGMENTED_DIR = Path("augmented")
STUDENTS_DIR  = Path("dataset/students")
OUTPUT_DIR    = Path("face_db")
OUTPUT_FILE   = OUTPUT_DIR / "embeddings.pkl"
WEIGHTS_FILE  = OUTPUT_DIR / "facenet_vggface2.pt"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_URL = (
    "https://github.com/timesler/facenet-pytorch"
    "/releases/download/v2.2.9/20180402-114759-vggface2.pt"
)

TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])




class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale   = scale
        self.branch0 = BasicConv2d(256, 32, 1, 1)
        self.branch1 = nn.Sequential(BasicConv2d(256, 32, 1, 1),
                                     BasicConv2d(32, 32, 3, 1, 1))
        self.branch2 = nn.Sequential(BasicConv2d(256, 32, 1, 1),
                                     BasicConv2d(32, 32, 3, 1, 1),
                                     BasicConv2d(32, 32, 3, 1, 1))
        self.conv2d  = nn.Conv2d(96, 256, 1)
        self.relu    = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat([self.branch0(x), self.branch1(x), self.branch2(x)], 1)
        return self.relu(self.scale * self.conv2d(out) + x)


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale   = scale
        self.branch0 = BasicConv2d(896, 128, 1, 1)
        self.branch1 = nn.Sequential(BasicConv2d(896, 128, 1, 1),
                                     BasicConv2d(128, 128, (1, 7), 1, (0, 3)),
                                     BasicConv2d(128, 128, (7, 1), 1, (3, 0)))
        self.conv2d  = nn.Conv2d(256, 896, 1)
        self.relu    = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat([self.branch0(x), self.branch1(x)], 1)
        return self.relu(self.scale * self.conv2d(out) + x)


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()
        self.scale   = scale
        self.noReLU  = noReLU
        self.branch0 = BasicConv2d(1792, 192, 1, 1)
        self.branch1 = nn.Sequential(BasicConv2d(1792, 192, 1, 1),
                                     BasicConv2d(192, 192, (1, 3), 1, (0, 1)),
                                     BasicConv2d(192, 192, (3, 1), 1, (1, 0)))
        self.conv2d  = nn.Conv2d(384, 1792, 1)
        if not noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat([self.branch0(x), self.branch1(x)], 1)
        out = self.scale * self.conv2d(out) + x
        return out if self.noReLU else self.relu(out)


class Mixed_6a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(256, 384, 3, 2)
        self.branch1 = nn.Sequential(BasicConv2d(256, 192, 1, 1),
                                     BasicConv2d(192, 192, 3, 1, 1),
                                     BasicConv2d(192, 256, 3, 2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x)], 1)


class Mixed_7a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(BasicConv2d(896, 256, 1, 1),
                                     BasicConv2d(256, 384, 3, 2))
        self.branch1 = nn.Sequential(BasicConv2d(896, 256, 1, 1),
                                     BasicConv2d(256, 256, 3, 2))
        self.branch2 = nn.Sequential(BasicConv2d(896, 256, 1, 1),
                                     BasicConv2d(256, 256, 3, 1, 1),
                                     BasicConv2d(256, 256, 3, 2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch0(x), self.branch1(x),
                          self.branch2(x), self.branch3(x)], 1)


class InceptionResnetV1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d_1a= BasicConv2d(3, 32, 3, 2)
        self.conv2d_2a= BasicConv2d(32, 32, 3, 1)
        self.conv2d_2b= BasicConv2d(32, 64, 3, 1, 1)
        self.maxpool_3a= nn.MaxPool2d(3, 2)
        self.conv2d_3b= BasicConv2d(64, 80, 1, 1)
        self.conv2d_4a= BasicConv2d(80, 192, 3, 1)
        self.conv2d_4b= BasicConv2d(192, 256, 3, 2)
        self.repeat_1= nn.Sequential(*[Block35(scale=0.17) for _ in range(5)])
        self.mixed_6a= Mixed_6a()
        self.repeat_2= nn.Sequential(*[Block17(scale=0.1) for _ in range(10)])
        self.mixed_7a= Mixed_7a()
        self.repeat_3= nn.Sequential(*[Block8(scale=0.2) for _ in range(5)])
        self.block8= Block8(noReLU=True)
        self.avgpool_1a= nn.AdaptiveAvgPool2d(1)
        self.dropout= nn.Dropout(0.6)
        self.last_linear= nn.Linear(1792, 512, bias=False)
        self.last_bn= nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.flatten(1))
        x = self.last_bn(x)
        return F.normalize(x, p=2, dim=1)



def download_weights():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if WEIGHTS_FILE.exists():
        print(f"Weights found:{WEIGHTS_FILE}")
        return True

    print(f"From: {WEIGHTS_URL}\n")
    try:
        r= requests.get(WEIGHTS_URL, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(WEIGHTS_FILE, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"\nSaved- {WEIGHTS_FILE}")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nDownload manually:")
        print(f"URL: {WEIGHTS_URL}")
        print(f"Save to: {WEIGHTS_FILE.resolve()}")
        return False

def load_model():
    if not download_weights():
        return None

    model= InceptionResnetV1().to(DEVICE)
    state_dict = torch.load(str(WEIGHTS_FILE), map_location=DEVICE, weights_only=False)

    # The .pt file stores a dict with key 'state_dict' or is a plain state_dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"FaceNet loaded, Device: {DEVICE}\n")
    return model

def get_embedding(model, img: Image.Image):
    try:
        tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return model(tensor).squeeze().cpu().numpy()
    except Exception:
        return None

def run():
    if AUGMENTED_DIR.exists() and any(AUGMENTED_DIR.iterdir()):
        source = AUGMENTED_DIR
        print("Using AUGMENTED dataset")
    elif STUDENTS_DIR.exists() and any(STUDENTS_DIR.iterdir()):
        source = STUDENTS_DIR
        print("Run augment.py first for better accuracy")
    else:
        print("No dataset found. Add photos to dataset/students/<Name>/")
        return

    folders = [f for f in source.iterdir() if f.is_dir()]
    if not folders:
        print(" No student folders found.")
        return

    model = load_model()
    if model is None:
        return

    print(f"{'='*55}")
    print(f"ENCODING — {len(folders)} students")
    print(f"{'='*55}\n")

    all_embeddings, all_names, failed = [], [], []

    for folder in folders:
        name   = folder.name
        images = list(folder.glob("*.jpg")) + \
                 list(folder.glob("*.jpeg")) + \
                 list(folder.glob("*.png"))
        if not images:
            print(f"SKIP {name} — no images")
            continue

        count = 0
        for img_path in tqdm(images, desc=f"  {name:25s}", leave=False):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            emb = get_embedding(model, img)
            if emb is not None:
                all_embeddings.append(emb)
                all_names.append(name)
                count += 1

        if count == 0:
            failed.append(name)
            print(f"{name:35s} No embeddings")
        else:
            print(f"{name:35s}  {count} embeddings")

    if not all_embeddings:
        print("\nNo embeddings created.")
        return

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"embeddings": all_embeddings, "names": all_names}, f)

    print(f"\n{'='*55}")
    print(f"DONE")
    print(f"Students: {len(set(all_names))}")
    print(f"Embeddings: {len(all_embeddings)}")
    print(f"Saved to: {OUTPUT_FILE}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()