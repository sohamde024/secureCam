
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

STUDENTS_DIR  = Path("dataset/students")
AUGMENTED_DIR = Path("augmented")


def augment_image(img: Image.Image):
    results = []
    w, h = img.size

    results.append(img.copy())
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    results.append(flipped)

    for factor in [0.3, 0.5, 0.65, 0.8, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0]:
        results.append(ImageEnhance.Brightness(img).enhance(factor))

        if factor in [0.5, 0.8, 1.3, 1.8]:
            results.append(ImageEnhance.Brightness(flipped).enhance(factor))

    for factor in [0.5, 0.7, 1.3, 1.6]:
        results.append(ImageEnhance.Contrast(img).enhance(factor))

    for angle in [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25]:
        results.append(img.rotate(angle, resample=Image.BICUBIC, expand=False))

    for radius in [1, 2, 3]:
        results.append(img.filter(ImageFilter.GaussianBlur(radius=radius)))

    for factor in [0.5, 2.0, 3.0]:
        results.append(ImageEnhance.Sharpness(img).enhance(factor))

    for margin_pct in [0.05, 0.10, 0.15]:
        mw, mh = int(w * margin_pct), int(h * margin_pct)
        cropped = img.crop((mw, mh, w - mw, h - mh))
        results.append(cropped.resize((w, h), Image.BICUBIC))

    gray = img.convert("L").convert("RGB")
    results.append(gray)
    # Gray + brightness combos
    for factor in [0.6, 1.4]:
        results.append(ImageEnhance.Brightness(gray).enhance(factor))

    for factor in [0.0, 0.5, 1.5, 2.0]:
        results.append(ImageEnhance.Color(img).enhance(factor))

    arr = np.array(img, dtype=np.int16)
    for sigma in [8, 15, 25]:
        noise = np.random.normal(0, sigma, arr.shape).astype(np.int16)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        results.append(Image.fromarray(noisy))
    return results


def run():
    if not STUDENTS_DIR.exists():
        print(f"\nFolder not found: {STUDENTS_DIR}")
        return

    folders = [f for f in STUDENTS_DIR.iterdir() if f.is_dir()]
    if not folders:
        print("No student folders found inside dataset/students/")
        return


    if AUGMENTED_DIR.exists():
        import shutil
        shutil.rmtree(AUGMENTED_DIR)
        print("Cleared old augmented data\n")

    print(f"\n{'='*50}")
    print(f"AUGMENTATION — {len(folders)} students")
    print(f"{'='*50}\n")

    total = 0
    for folder in folders:
        name    = folder.name
        out_dir = AUGMENTED_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = list(folder.glob("*.jpg")) + \
                 list(folder.glob("*.jpeg")) + \
                 list(folder.glob("*.png"))

        if not images:
            print(f"{name} — no images found")
            continue

        count = 0
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224), Image.BICUBIC)
            except Exception:
                print(f"Could not read {img_path.name}")
                continue

            for aug in augment_image(img):
                aug = aug.resize((224, 224), Image.BICUBIC)
                aug.save(str(out_dir / f"{img_path.stem}_aug{count:03d}.jpg"),
                         quality=95)
                count += 1

        total += count
        print(f"{name:30s} -> {count} images")

    print(f"\nDONE— {total} total images")
    print(f"{'='*50}")


if __name__ == "__main__":
    run()