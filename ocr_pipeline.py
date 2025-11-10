# ocr_pipeline.py
"""
Importable OCR pipeline for UI.

Provides:
    process_image(image_path_or_array, weights_path="models/best.pt", conf=0.25,
                  save_crops_dir="results/crops", tesseract_psm="6", tesseract_lang="eng")

Saves annotated images and crops to results/ and returns a pandas.DataFrame.
"""

from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from ultralytics import YOLO

# If you're on Windows and Tesseract is installed in Program Files, uncomment and set path:
# import platform
# if platform.system() == "Windows":
#     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_for_ocr(crop, scale=3):
    """Resize, grayscale, blur, Otsu-threshold, ensure dark text on light bg."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    new_w = max(32, int(w * scale))
    new_h = max(32, int(h * scale))
    try:
        img = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        img = crop.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure dark text on light background for tesseract
    if th.mean() > 127:
        th = cv2.bitwise_not(th)
    return th

def _ocr_from_image_array(img, model, save_crops_dir="results/crops", image_name="image",
                          conf=0.25, tesseract_psm="6", tesseract_lang="eng"):
    """
    Run YOLO detection on a BGR numpy image and OCR each crop.
    Returns list of dict rows with keys: filename,xmin,ymin,xmax,ymax,conf,class,text,crop_path,annotated_path
    """
    save_crops_dir = Path(save_crops_dir)
    save_crops_dir.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    # run detection: try both common ultralytics call patterns
    try:
        results = model.predict(source=img, conf=conf, verbose=False)
        res = results[0]
    except Exception:
        results = model(img, conf=conf, verbose=False)
        res = results[0]

    det = getattr(res, "boxes", None)
    rows = []
    ann = img.copy()

    # Fallback: no detections -> OCR whole image
    if det is None or len(det) == 0:
        pre = preprocess_for_ocr(img)
        text = ""
        if pre is not None:
            try:
                pil_img = Image.fromarray(pre)
                text = pytesseract.image_to_string(pil_img, lang=tesseract_lang, config=f"--psm {tesseract_psm}")
            except Exception:
                text = ""
        crop_path = save_crops_dir / f"{Path(image_name).stem}_full.png"
        if pre is not None:
            cv2.imwrite(str(crop_path), pre)
        ann_path = Path("results") / f"annotated_{Path(image_name).stem}.png"
        cv2.imwrite(str(ann_path), ann)
        rows.append({
            "filename": image_name,
            "xmin": 0, "ymin": 0, "xmax": img.shape[1], "ymax": img.shape[0],
            "conf": 0.0, "class": "full_image", "text": (text or "").strip(),
            "crop_path": str(crop_path), "annotated_path": str(ann_path)
        })
        return rows

    # Iterate detections
    for i, box in enumerate(det):
        # get bbox coords
        try:
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else box.xyxy
        except Exception:
            xyxy = box.xyxy
        try:
            confv = float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else float(getattr(box, 'conf', 0.0))
        except Exception:
            confv = float(getattr(box, 'conf', 0.0))
        try:
            cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else int(getattr(box, 'cls', -1))
        except Exception:
            cls = int(getattr(box, 'cls', -1))

        x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        # annotate on ann image
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(ann, f"{confv:.2f}", (x1, max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

        crop = img[y1:y2, x1:x2]
        pre = preprocess_for_ocr(crop)
        text = ""
        if pre is not None:
            try:
                pil_img = Image.fromarray(pre)
                text = pytesseract.image_to_string(pil_img, lang=tesseract_lang, config=f"--psm {tesseract_psm}")
            except Exception:
                text = ""
        crop_path = save_crops_dir / f"{Path(image_name).stem}_crop_{i}.png"
        if pre is not None:
            cv2.imwrite(str(crop_path), pre)

        rows.append({
            "filename": image_name,
            "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
            "conf": confv, "class": cls, "text": (text or "").strip(),
            "crop_path": str(crop_path)
        })

    # save annotated image and attach path to rows
    ann_path = Path("results") / f"annotated_{Path(image_name).stem}.png"
    cv2.imwrite(str(ann_path), ann)
    for r in rows:
        r["annotated_path"] = str(ann_path)

    return rows

def process_image(image_path_or_array, weights_path="models/best.pt", conf=0.25,
                  save_crops_dir="results/crops", tesseract_psm="6", tesseract_lang="eng"):
    """
    Public function for UI:
      - image_path_or_array: path (str/Path) or numpy BGR array
      - weights_path: path to YOLO .pt
      - conf: detection confidence
      - save_crops_dir: where to save crops
      - tesseract_psm: PSM string for Tesseract
      - tesseract_lang: Tesseract language(s)
    Returns: pandas.DataFrame of rows (may be empty)
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found at: {weights_path}")

    # load model
    model = YOLO(str(weights_path))

    # load image array
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
        image_name = Path(image_path_or_array).name
    else:
        img = image_path_or_array
        image_name = "uploaded_image"

    if img is None:
        raise ValueError("Could not read image (img is None)")

    start = time.time()
    rows = _ocr_from_image_array(img, model=model, save_crops_dir=save_crops_dir,
                                image_name=image_name, conf=conf,
                                tesseract_psm=str(tesseract_psm), tesseract_lang=str(tesseract_lang))
    elapsed = time.time() - start

    df = pd.DataFrame(rows)
    out_csv = Path("results") / f"{Path(image_name).stem}_ocr.csv"
    df.to_csv(out_csv, index=False)
    if not df.empty:
        df["runtime_s"] = elapsed
    return df

# CLI fallback
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--out", default="results/ocr_output.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--psm", default="6")
    parser.add_argument("--lang", default="eng")
    args = parser.parse_args()
    df = process_image(args.image, weights_path=args.weights, conf=args.conf,
                       save_crops_dir="results/crops", tesseract_psm=args.psm, tesseract_lang=args.lang)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
