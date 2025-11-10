# ui_app_polished.py
"""
Polished, animated Streamlit UI for Custom OCR — styled like NewsMatrix Pro.
Requires:
  - ocr_pipeline.py in same folder exposing process_image(...)
  - streamlit, ultralytics, pytesseract, opencv-python, pillow, pandas, numpy installed
Run:
  streamlit run ui_app_polished.py
"""

import streamlit as st
from pathlib import Path
import tempfile, io, time, zipfile
from PIL import Image
import numpy as np
import pandas as pd
import cv2

# Try to import the pipeline's process_image function
try:
    from ocr_pipeline import process_image
except Exception as e:
    process_image = None
    import traceback
    import sys
    tb = traceback.format_exc()

# Page + Animated Gradient + Glass CSS (inspired by your NewsMatrix design)
st.set_page_config(page_title="✨ OCRMatrix Pro", page_icon="🧾", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(-45deg, #0f172a, #1e293b, #0ea5e9, #7c3aed);
            background-size: 400% 400%;
            animation: gradient 18s ease infinite;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #e6eef8;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .neo-glass {
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(10px) saturate(120%);
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.06);
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 8px 30px rgba(2,6,23,0.5);
            transition: transform .25s ease, box-shadow .25s ease;
        }
        .neo-glass:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(2,6,23,0.6); }
        .logo {
            width:54px; height:54px; border-radius:12px; display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg,#06b6d4,#7c3aed); color:#021124; font-weight:800;
            box-shadow: 0 8px 30px rgba(2,6,23,0.4);
        }
        .title { font-size:28px; font-weight:700; margin:0; color: #f8fafc; }
        .subtitle { margin:0; color: rgba(236,239,244,0.85); }
        .small-muted { color: rgba(236,239,244,0.6); font-size:13px; }
        .premium-card { padding:12px; border-radius:12px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.03); }
        .stButton button { border-radius: 999px; padding: .6rem 1.6rem; font-weight:700; }
        .crop-img { border-radius:8px; box-shadow: 0 8px 28px rgba(2,6,23,0.45); }
        .footer-note { color: rgba(236,239,244,0.55); font-size:13px; margin-top:12px; }
    </style>
    """, unsafe_allow_html=True)

# Header
c1, c2 = st.columns([0.09, 0.91])
with c1:
    st.markdown('<div class="logo">OCR</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div style="padding-left:10px;"><h1 class="title">OCRMatrix Pro</h1><div class="subtitle">Custom OCR • YOLO + Tesseract — polished & shareable demo</div></div>', unsafe_allow_html=True)

# Sidebar controls (like your NewsMatrix)
with st.sidebar:
    st.markdown("<div class='neo-glass'><h3 style='margin:4px 0;'>⚙️ Controls</h3>", unsafe_allow_html=True)
    weights_file = st.file_uploader("Upload YOLO weights (.pt) (optional)", type=["pt"])
    conf = st.slider("Detection confidence", 0.01, 1.0, 0.25, step=0.01)
    tesseract_psm = st.selectbox("Tesseract PSM", options=["3","4","6","11"], index=2)
    tesseract_lang = st.text_input("Tesseract language(s)", value="eng")
    show_crops = st.checkbox("Show crop previews", value=True)
    save_results = st.checkbox("Save results to results/", value=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='neo-glass'><h4 style='margin:4px 0;'>Quick Actions</h4>", unsafe_allow_html=True)
    if st.button("Clear results folder"):
        # clear results safely
        for f in Path("results").glob("*"):
            try:
                if f.is_dir():
                    for g in f.rglob("*"):
                        if g.is_file(): g.unlink()
                    f.rmdir()
                else:
                    f.unlink()
            except Exception:
                pass
        st.success("Cleared results/ (if existed)")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;' class='neo-glass'><small class='small-muted'>Tip: Set confidence low (0.01) for debugging, then raise it for production.</small></div>", unsafe_allow_html=True)

# main upload area (card)
st.markdown("<div class='neo-glass'>", unsafe_allow_html=True)
uploaded_images = st.file_uploader("Upload images (png/jpg) — drag & drop multiple", type=["png","jpg","jpeg"], accept_multiple_files=True)
st.markdown("</div>", unsafe_allow_html=True)

# handle weights upload -> temp file
if weights_file is not None:
    tmpw = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmpw.write(weights_file.read())
    tmpw.flush()
    weights_path = tmpw.name
    st.sidebar.success("Weights uploaded (temporary)")
else:
    weights_path = "models/best.pt"  # default

# Quick action run button
run_col, debug_col = st.columns([0.85, 0.15])
with run_col:
    run_btn = st.button("🚀 Run OCR", key="run_btn")
with debug_col:
    debug_toggle = st.checkbox("Debug", value=False)

# Show import error if pipeline missing
if process_image is None:
    st.error("ocr_pipeline.process_image not available. Put updated ocr_pipeline.py (with process_image) in the same folder.")
    if 'tb' in globals():
        st.code(tb)
    st.stop()

# Ensure results dirs exist
Path("results").mkdir(parents=True, exist_ok=True)
Path("results/crops").mkdir(parents=True, exist_ok=True)

# On Run: process each image with nice layout
if run_btn:
    if not uploaded_images:
        st.warning("Upload at least one image.")
    else:
        all_dfs = []
        timings = []
        for file in uploaded_images:
            filename = getattr(file, "name", f"uploaded_{int(time.time())}.jpg")
            st.markdown(f"<div class='neo-glass'><strong>Processing:</strong> {filename}</div>", unsafe_allow_html=True)
            try:
                img_bytes = file.read()
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                st.error(f"Could not read {filename}: {e}")
                continue

            np_bgr = np.array(pil)[:, :, ::-1].copy()
            t0 = time.time()
            try:
                # call pipeline
                df = process_image(np_bgr,
                                   weights_path=weights_path,
                                   conf=float(conf),
                                   save_crops_dir="results/crops",
                                   tesseract_psm=str(tesseract_psm),
                                   tesseract_lang=str(tesseract_lang))
            except Exception as e:
                st.error(f"Pipeline error for {filename}: {e}")
                if debug_toggle:
                    st.exception(e)
                continue
            elapsed = time.time() - t0
            timings.append((filename, elapsed))
            st.markdown(f"<div class='small-muted'>Time: {elapsed:.2f}s</div>", unsafe_allow_html=True)

            # Annotated image: draw boxes on PIL for display
            def draw_boxes_on_pil(pil_img, df_local):
                if df_local is None or df_local.empty:
                    return pil_img
                img_arr = np.array(pil_img.convert("RGB"))
                bgr = img_arr[:, :, ::-1].copy()
                for _, row in df_local.iterrows():
                    try:
                        x1,y1,x2,y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        confv = float(row.get('conf', 0.0))
                    except Exception:
                        continue
                    cv2.rectangle(bgr, (x1,y1), (x2,y2), (34,197,94), 2)
                    cv2.putText(bgr, f"{confv:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34,197,94), 1, cv2.LINE_AA)
                return Image.fromarray(bgr[:, :, ::-1])

            annotated = draw_boxes_on_pil(pil, df)

            # show original + annotated side-by-side
            colA, colB = st.columns([1,1])
            with colA:
                st.image(pil, caption=f"Original — {filename}", use_container_width=True)
            with colB:
                st.image(annotated, caption="Annotated", use_container_width=True)

            # show crop previews saved in results/crops (matching filename prefix)
            if show_crops:
                prefix = Path(filename).stem
                crops = sorted(Path("results/crops").glob(f"{prefix}*"))
                if crops:
                    st.markdown("<div class='neo-glass'><strong>Crops</strong></div>", unsafe_allow_html=True)
                    crop_cols = st.columns(3)
                    for i, c in enumerate(crops[:9]):
                        try:
                            imgc = Image.open(c)
                            crop_cols[i % 3].image(imgc, use_container_width=True, caption=c.name)
                        except Exception:
                            continue
                else:
                    st.markdown("<div class='small-muted'>No crops saved (check detections/confidence).</div>", unsafe_allow_html=True)

            # Editable table (allow correction)
            st.markdown("<div class='neo-glass'><strong>Detected text (editable)</strong></div>", unsafe_allow_html=True)
            display_df = df.copy() if df is not None else pd.DataFrame()
            preferred = [c for c in ["xmin","ymin","xmax","ymax","conf","text"] if c in display_df.columns]
            if preferred:
                display_df = display_df[preferred + [c for c in display_df.columns if c not in preferred]]
            try:
                edited = st.data_editor(display_df, num_rows="dynamic", key=f"editor_{filename}")
            except Exception:
                st.dataframe(display_df)
                edited = display_df

            # Save per-image CSV if requested
            if save_results:
                out_csv = Path("results") / f"{Path(filename).stem}_ocr.csv"
                pd.DataFrame(edited).to_csv(out_csv, index=False)
                st.success(f"Saved: {out_csv}")

            all_dfs.append(pd.DataFrame(edited))

        # Post-process combined results + downloads
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined_path = Path("results") / "combined_ocr.csv"
            combined.to_csv(combined_path, index=False)

            st.markdown("<div class='neo-glass'><strong>Downloads</strong></div>", unsafe_allow_html=True)
            with open(combined_path, "rb") as f:
                st.download_button("Download combined CSV", f.read(), file_name=combined_path.name, mime="text/csv")
            zip_out = Path("results") / "results_bundle.zip"
            with zipfile.ZipFile(zip_out, "w") as zf:
                for f in Path("results").rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(Path("results")))
            with open(zip_out, "rb") as f:
                st.download_button("Download results ZIP", f.read(), file_name=zip_out.name, mime="application/zip")

            st.markdown("<div class='neo-glass'><strong>Summary</strong></div>", unsafe_allow_html=True)
            times_df = pd.DataFrame(timings, columns=["filename","seconds"])
            st.table(times_df)

            st.balloons()
        else:
            st.warning("No results produced (check model, weights, or conf param).")

# Footer help
st.markdown("<div class='footer-note'>Tips: If OCR returns empty text on Windows, set tesseract path in ocr_pipeline.py: pytesseract.pytesseract.tesseract_cmd = r\"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\". Lower confidence to 0.01 to debug detections.</div>", unsafe_allow_html=True)
