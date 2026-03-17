import pickle
import time
import datetime
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2


EMBEDDINGS_FILE    = Path("face_db/embeddings.pkl")
WEIGHTS_FILE       = Path("face_db/facenet_vggface2.pt")
MP_MODEL_FILE      = Path("face_db/blaze_face_short_range.tflite")
MP_MODEL_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
THRESHOLD          = 0.67
FACE_PAD           = 50
ALERT_COOLDOWN_SEC = 5
SCREENSHOTS_DIR    = Path("screenshots")
LOG_FILE           = Path("detection_log.txt")
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

GREEN = (0, 210, 80)
RED   = (255, 50, 50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (150, 150, 150)



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
        self.branch1 = nn.Sequential(BasicConv2d(256,32,1,1), BasicConv2d(32,32,3,1,1))
        self.branch2 = nn.Sequential(BasicConv2d(256,32,1,1), BasicConv2d(32,32,3,1,1), BasicConv2d(32,32,3,1,1))
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
        self.branch1 = nn.Sequential(BasicConv2d(896,128,1,1), BasicConv2d(128,128,(1,7),1,(0,3)), BasicConv2d(128,128,(7,1),1,(3,0)))
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
        self.branch1 = nn.Sequential(BasicConv2d(1792,192,1,1), BasicConv2d(192,192,(1,3),1,(0,1)), BasicConv2d(192,192,(3,1),1,(1,0)))
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
        self.branch1 = nn.Sequential(BasicConv2d(256,192,1,1), BasicConv2d(192,192,3,1,1), BasicConv2d(192,256,3,2))
        self.branch2 = nn.MaxPool2d(3, stride=2)
    def forward(self, x):
        return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x)], 1)

class Mixed_7a(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(BasicConv2d(896,256,1,1), BasicConv2d(256,384,3,2))
        self.branch1 = nn.Sequential(BasicConv2d(896,256,1,1), BasicConv2d(256,256,3,2))
        self.branch2 = nn.Sequential(BasicConv2d(896,256,1,1), BasicConv2d(256,256,3,1,1), BasicConv2d(256,256,3,2))
        self.branch3 = nn.MaxPool2d(3, stride=2)
    def forward(self, x):
        return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)], 1)

class InceptionResnetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1a   = BasicConv2d(3,32,3,2);    self.conv2d_2a  = BasicConv2d(32,32,3,1)
        self.conv2d_2b   = BasicConv2d(32,64,3,1,1); self.maxpool_3a = nn.MaxPool2d(3,2)
        self.conv2d_3b   = BasicConv2d(64,80,1,1);   self.conv2d_4a  = BasicConv2d(80,192,3,1)
        self.conv2d_4b   = BasicConv2d(192,256,3,2)
        self.repeat_1    = nn.Sequential(*[Block35(0.17) for _ in range(5)])
        self.mixed_6a    = Mixed_6a()
        self.repeat_2    = nn.Sequential(*[Block17(0.1) for _ in range(10)])
        self.mixed_7a    = Mixed_7a()
        self.repeat_3    = nn.Sequential(*[Block8(0.2) for _ in range(5)])
        self.block8      = Block8(noReLU=True)
        self.avgpool_1a  = nn.AdaptiveAvgPool2d(1)
        self.dropout     = nn.Dropout(0.6)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn     = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.conv2d_1a(x);  x = self.conv2d_2a(x);  x = self.conv2d_2b(x)
        x = self.maxpool_3a(x); x = self.conv2d_3b(x);  x = self.conv2d_4a(x)
        x = self.conv2d_4b(x);  x = self.repeat_1(x);   x = self.mixed_6a(x)
        x = self.repeat_2(x);   x = self.mixed_7a(x);   x = self.repeat_3(x)
        x = self.block8(x);     x = self.avgpool_1a(x); x = self.dropout(x)
        x = self.last_linear(x.flatten(1))
        x = self.last_bn(x)
        return F.normalize(x, p=2, dim=1)



def download_file(url, dest: Path, label="Downloading"):
    if dest.exists():
        print(f"[INFO] Found: {dest.name}")
        return True
    dest.parent.mkdir(exist_ok=True)
    print(f"[INFO] {label} → {dest.name}")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"[INFO] Saved: {dest}")
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def load_facenet():
    if not WEIGHTS_FILE.exists():
        print(f"{WEIGHTS_FILE} not found")
        return None
    model= InceptionResnetV1().to(DEVICE)
    state_dict = torch.load(str(WEIGHTS_FILE), map_location=DEVICE, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"FaceNet loaded, Device: {DEVICE}")
    return model


def load_database():
    if not EMBEDDINGS_FILE.exists():
        print("face_db/embeddings.pkl not found")
        return None, None
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"{len(data['embeddings'])} embeddings{len(set(data['names']))} students")
    return data["embeddings"], data["names"]


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def find_match(probe, db_embs, db_names):
    sims = [cosine_sim(probe, e) for e in db_embs]
    idx = int(np.argmax(sims))
    return db_names[idx], sims[idx]


def get_embedding(model, face_pil: Image.Image):
    try:
        t = TRANSFORM(face_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return model(t).squeeze().cpu().numpy()
    except Exception:
        return None


def deduplicate_boxes(boxes, iou_threshold=0.4):

    if len(boxes) <= 1:
        return boxes
    kept = []
    for box in boxes:
        x1, y1, x2, y2 = box
        is_dup = False
        for kx1, ky1, kx2, ky2 in kept:
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (kx2 - kx1) * (ky2 - ky1)
            iou = inter / (area1 + area2 - inter + 1e-6)
            if iou > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(box)
    return kept



class FaceDetectorWrapper:


    def __init__(self):
        self.mp_det  = None   # holds the mediapipe FaceDetector instance
        self.mode    = None
        self._setup()

    def _setup(self):
        ok = download_file(MP_MODEL_URL, MP_MODEL_FILE, "Downloading MediaPipe model")
        if not ok:
            self.mode = "fullframe"
            print("Using full-frame fallback")
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision import (
                FaceDetector as MpFaceDetector,
                FaceDetectorOptions,
            )
            from mediapipe.tasks.python.core.base_options import BaseOptions

            options  = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(MP_MODEL_FILE)),
                min_detection_confidence=0.4,
                min_suppression_threshold=0.5,
            )
            self.mp_det = MpFaceDetector.create_from_options(options)
            self.mode   = "tasks"
            print("MediaPipe Tasks API ready — multi-face detection ON")

        except Exception as e:
            self.mode = "fullframe"
            print(f"MediaPipe setup failed ({e})full-frame fallback")

    def run_detection(self, pil_img: Image.Image):

        W, H  = pil_img.size
        boxes = []

        if self.mode != "tasks":
            return [(0, 0, W, H)]

        import mediapipe as mp

        def detect_on_tile(tile_pil, offset_x=0, offset_y=0):

            tile_np  = np.array(tile_pil.convert("RGB"), dtype=np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=tile_np)
            try:
                result = self.mp_det.detect(mp_image)
            except Exception as e:
                print(f"Detection Error: {e}")
                return []
            tile_boxes = []
            for det in result.detections:
                bb = det.bounding_box
                x1 = max(0,  offset_x + bb.origin_x - FACE_PAD)
                y1 = max(0,  offset_y + bb.origin_y - FACE_PAD)
                x2 = min(W,  offset_x + bb.origin_x + bb.width  + FACE_PAD)
                y2 = min(H,  offset_y + bb.origin_y + bb.height + FACE_PAD)
                tile_boxes.append((x1, y1, x2, y2))
            return tile_boxes


        boxes += detect_on_tile(pil_img)


        if W >= 640:
            half_w = W // 2
            # Left half
            left_tile  = pil_img.crop((0, 0, half_w, H))
            boxes += detect_on_tile(left_tile, offset_x=0)
            # Right half
            right_tile = pil_img.crop((half_w, 0, W, H))
            boxes += detect_on_tile(right_tile, offset_x=half_w)


        boxes = deduplicate_boxes(boxes)
        return boxes



def get_font(size=14):
    try: return ImageFont.truetype("arial.ttf", size)
    except: return ImageFont.load_default()


def draw_face_box(draw, x1, y1, x2, y2, name, sim, is_student):
    color = GREEN if is_student else RED
    L = 18
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        draw.line([(px,py),(px+dx*L,py)], fill=color, width=3)
        draw.line([(px,py),(px,py+dy*L)], fill=color, width=3)
    label = f" {name.replace('_',' ')}  {sim*100:.0f}% " if is_student \
            else " UNKNOWN — ALERT! "
    font = get_font(14)
    bb = draw.textbbox((0,0), label, font=font)
    tw,th = bb[2]-bb[0], bb[3]-bb[1]
    ly = max(y1-th-8, 0)
    draw.rectangle([x1, ly-2, x1+tw+4, ly+th+4], fill=color)
    draw.text((x1+2, ly), label, fill=BLACK if is_student else WHITE, font=font)


def draw_hud(draw, iw, ih, fps, verified, alerts, alert_active, mode, face_count):
    fsm = get_font(13); fmd = get_font(16)
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    draw.text((10, 8),  f"GateGuard AI  |  {now}  |  {mode}", fill=GRAY, font=fsm)
    draw.text((10, 26), f"FPS:{fps:.1f}  Faces:{face_count}  Verified:{verified}  Alerts:{alerts}",
              fill=GRAY, font=fsm)
    draw.rectangle([0, ih-36, iw, ih],
                   fill=(0,55,0) if not alert_active else (140,0,0))
    msg = "!!UNAUTHORIZED PERSON — ALERTING SECURITY!!" if alert_active \
          else "ALL CLEAR — All persons verified as college students"
    draw.text((10, ih-26), msg,
              fill=WHITE if alert_active else GREEN, font=fmd)
    draw.text((iw-250, ih-22), "Q:Quit | S:Screenshot", fill=(90,90,90), font=fsm)


def send_alert(frame_pil: Image.Image):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'!'*55}")
    print(f"SECURITY ALERT — {now}")
    print(f"UNKNOWN PERSON detected at gate!")
    print(f"Notifying: Gate Keeper | Security Management Team")
    print(f"{'!'*55}\n")
    try:
        import winsound
        winsound.Beep(1000, 350)
        winsound.Beep(700,  350)
    except Exception:
        print("\a")
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_pil.save(str(SCREENSHOTS_DIR / f"ALERT_{ts}.jpg"))
    print(f"Screenshot saved -> ALERT_{ts}.jpg\n")


def log_event(name, sim, is_student):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{now}]  {'STUDENT' if is_student else 'UNKNOWN':10s}  "
                f"{name:30s}  sim={sim:.4f}\n")



def run():
    db_embs, db_names = load_database()
    if db_embs is None:
        return

    facenet = load_facenet()
    if facenet is None:
        return

    detector = FaceDetectorWrapper()   # our wrapper — no naming conflict

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("\n[LIVE] Camera started — press Q to quit\n")

    fps = 0.0
    prev_time = time.time()
    verified_count = 0
    alert_count = 0
    last_alert_time = 0
    frame_idx = 0
    last_results = []

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame = Image.fromarray(bgr[:, :, ::-1])   # BGR → RGB
        iw, ih = frame.size


        if frame_idx % 2 == 0:
            boxes = detector.run_detection(frame)   # use run_detection()
            last_results = []

            for (x1, y1, x2, y2) in boxes:
                crop = frame.crop((x1, y1, x2, y2))
                if crop.size[0] < 20 or crop.size[1] < 20:
                    continue
                emb = get_embedding(facenet, crop)
                if emb is None:
                    last_results.append((x1, y1, x2, y2, "UNKNOWN", 0.0, False))
                    continue
                name, sim = find_match(emb, db_embs, db_names)
                is_student = sim >= THRESHOLD
                last_results.append((x1, y1, x2, y2, name, sim, is_student))

        # Draw results
        draw = ImageDraw.Draw(frame)
        alert_this_frame = False

        for (x1, y1, x2, y2, name, sim, is_student) in last_results:
            draw_face_box(draw, x1, y1, x2, y2, name, sim, is_student)
            log_event(name, sim, is_student)
            if is_student:
                verified_count += 1
            else:
                alert_this_frame = True
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN_SEC:
                    alert_count += 1
                    last_alert_time = now
                    send_alert(frame)

        now_t = time.time()
        fps = 0.9 * fps + 0.1 / max(now_t - prev_time, 1e-5)
        prev_time = now_t
        draw_hud(draw, iw, ih, fps, verified_count, alert_count,
                 alert_this_frame, detector.mode, len(last_results))

        # Convert PIL → BGR for cv2 display
        cv2.imshow("GateGuard AI (Q to quit)", np.array(frame)[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            SCREENSHOTS_DIR.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            frame.save(str(SCREENSHOTS_DIR / f"capture_{ts}.jpg"))
            print("[SAVED] Screenshot saved.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE]Verified:{verified_count} Alerts:{alert_count}")
    print(f"Log -> {LOG_FILE}")


if __name__ == "__main__":
    run()