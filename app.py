"""SatReCIL satellite image classification dashboard."""
from __future__ import annotations

import io
import math
import time
from pathlib import Path

import streamlit as st
import torch
import yaml
from PIL import Image, UnidentifiedImageError

from scripts.predict_image import load_model, preprocess_image


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
try:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        CONFIG = yaml.safe_load(config_file) or {}
    CONFIG_ERROR = None
except (OSError, yaml.YAMLError) as exc:
    CONFIG = {}
    CONFIG_ERROR = str(exc)

SEED = CONFIG.get("dataset", {}).get("seed", 42)
STAGE_CLASSES = CONFIG.get("dataset", {}).get("stage_classes", {})
EXPECTED_CLASSES = STAGE_CLASSES.get("stage3", [])
ICONS = {"Forest": "🌲", "River": "🌊", "Highway": "🛣️", "Residential": "🏘️", "Industrial": "🏭"}
STAGE_INFO = {
    1: ("Stage 1 — Initial Learning", "base"),
    2: ("Stage 2 — Incremental Learning: Highway", "replay"),
    3: ("Stage 3 — Incremental Learning: Industrial", "replay"),
}

st.set_page_config(page_title="SatReCIL — Satellite Scene Intelligence", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 20% 0%, #142536 0, #0a111b 45%, #080d14 100%); color:#e8eef5}
.block-container {max-width:1200px;padding-top:2.2rem}
h1,h2,h3 {letter-spacing:-.03em}
.hero {padding:1.4rem 0 1.2rem;border-bottom:1px solid #253342;margin-bottom:1.5rem}
.eyebrow {color:#63d5c5;text-transform:uppercase;letter-spacing:.16em;font-size:.76rem;font-weight:700}
.panel {background:#101b27;border:1px solid #263746;border-radius:18px;padding:1.25rem}
.muted {color:#96a8b8}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">Remote sensing · Incremental AI</div><h1>SatReCIL — Satellite Scene Intelligence</h1><div class="muted">Incremental Learning for Satellite Image Classification</div></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_stage_model(stage, checkpoint_path):
    """Load and validate one stage's own cached checkpoint and output mapping."""
    stage_key = f"stage{stage}"
    expected_classes = STAGE_CLASSES.get(stage_key)
    if not expected_classes:
        raise ValueError(f"{CONFIG_PATH.name} has no {stage_key} class list.")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"{stage_key} checkpoint is missing: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, saved_stage, classes, class_to_idx, _ = load_model(Path(checkpoint_path), device)
    expected_mapping = {class_name: index for index, class_name in enumerate(expected_classes)}
    state_dict = checkpoint.get("model_state_dict", {})
    classifier_weight = state_dict.get("classifier.weight")

    if saved_stage != stage:
        raise ValueError(f"Expected a Stage {stage} checkpoint, but metadata says Stage {saved_stage}.")
    if checkpoint.get("stage_classes") != expected_classes or classes != expected_classes:
        raise ValueError(
            f"{stage_key} checkpoint class order {classes} does not match config.yaml: {expected_classes}."
        )
    if checkpoint.get("num_classes") != len(expected_classes) or model.num_classes != len(expected_classes):
        raise ValueError(f"{stage_key} checkpoint output dimension does not match its {len(expected_classes)} configured classes.")
    if classifier_weight is None or classifier_weight.shape[0] != len(expected_classes):
        raise ValueError(f"{stage_key} checkpoint classifier weights have an unexpected output dimension.")
    if class_to_idx != expected_mapping:
        raise ValueError(f"{stage_key} checkpoint class_to_idx does not match configured output order: {expected_mapping}.")
    return model, device, expected_classes


def analyze(uploaded):
    raw = uploaded.getvalue()
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file is not a valid JPG, JPEG, or PNG image.") from exc
    width, height = image.size
    # Existing CLI preprocessing accepts a path; use a temporary file only during inference.
    import tempfile
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        image.save(temp.name, format="PNG")
        _, tensor, _ = preprocess_image(Path(temp.name))
    if tuple(tensor.shape) != (1, 3, 64, 64):
        raise ValueError(f"Preprocessing produced unexpected input shape {tuple(tensor.shape)}; expected (1, 3, 64, 64).")

    stage_results = {}
    total_start = time.perf_counter()
    for stage in (1, 2, 3):
        title, method = STAGE_INFO[stage]
        checkpoint_path = ROOT / "checkpoints" / f"stage{stage}_{method}_seed{SEED}.pt"
        try:
            model, device, classes = get_stage_model(stage, str(checkpoint_path))
            start = time.perf_counter()
            with torch.inference_mode():
                logits = model(tensor.to(device))
                if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(classes):
                    raise ValueError(
                        f"Model returned shape {tuple(logits.shape)}; expected (1, {len(classes)})."
                    )
                probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
            if len(probabilities) != len(classes) or not all(math.isfinite(value) for value in probabilities):
                raise ValueError("Model produced an invalid probability distribution.")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Model probabilities do not sum to 1.")
            predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            stage_results[stage] = {
                "title": title,
                "checkpoint": checkpoint_path.name,
                "classes": classes,
                "prediction": classes[predicted_index],
                "confidence": probabilities[predicted_index],
                "probabilities": probabilities,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            }
        except Exception as exc:
            stage_results[stage] = {
                "title": title,
                "checkpoint": checkpoint_path.name,
                "classes": STAGE_CLASSES.get(f"stage{stage}", []),
                "error": str(exc),
            }

    return {
        "name": uploaded.name,
        "image": image.copy(),
        "width": width,
        "height": height,
        "stages": stage_results,
        "elapsed_ms": (time.perf_counter() - total_start) * 1000,
    }


if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "history" not in st.session_state:
    st.session_state.history = []
else:
    st.session_state.history = [
        item for item in st.session_state.history
        if isinstance(item, dict) and isinstance(item.get("stages"), dict)
    ]

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Analyze a scene")
    uploaded = st.file_uploader("Drop a satellite image or browse", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key=f"upload_{st.session_state.upload_key}")
    if uploaded:
        try:
            preview = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            st.image(preview, use_container_width=True)
            st.caption(f"{uploaded.name} · {preview.width} × {preview.height} px")
            if st.button("Analyze Image", type="primary", use_container_width=True):
                try:
                    result = analyze(uploaded)
                    st.session_state.history.insert(0, result)
                except Exception as exc:
                    st.error(f"Could not analyze this image: {exc}")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(str(exc))
        if st.button("Remove image"):
            st.session_state.upload_key += 1
            st.rerun()
    else:
        st.info("Choose a JPG, JPEG, or PNG image to begin.")

with right:
    st.subheader("Analysis")
    if st.session_state.history:
        result = st.session_state.history[0]
        st.success(f"Completed stage comparison for **{result['name']}**.")
        st.caption(f"Processing time across three stage checkpoints: {result['elapsed_ms']:.1f} ms")
    else:
        st.info("Upload an image, then analyze it with each stage's own checkpoint.")

if st.session_state.history:
    st.divider()
    history_title, history_button = st.columns([3, 1])
    history_title.subheader("Session history")
    if history_button.button("Clear history"):
        st.session_state.history.clear()
        st.rerun()

    if len(st.session_state.history) > 1:
        with st.expander(f"Earlier image analyses ({len(st.session_state.history) - 1})"):
            for previous in st.session_state.history[1:]:
                predictions = [
                    f"S{stage}: {record.get('prediction', 'Unavailable')}"
                    for stage, record in previous.get("stages", {}).items()
                ]
                st.write(f"**{previous['name']}** · " + " · ".join(predictions))

if CONFIG_ERROR:
    st.error(f"Could not read the project configuration: {CONFIG_ERROR}")

if st.session_state.history:
    result = st.session_state.history[0]
    st.divider()
    st.header("TerraCIL Predictions Across Learning Stages")
    st.caption(f"Independent predictions for {result['name']} · uploaded image preview is shown above.")
    columns = st.columns(3)
    for stage, column in zip((1, 2, 3), columns):
        prediction = result.get("stages", {}).get(stage, {})
        classes = prediction.get("classes", STAGE_CLASSES.get(f"stage{stage}", []))
        with column.container(border=True):
            st.subheader(prediction.get("title", STAGE_INFO[stage][0]))
            st.caption(f"{len(classes)} classes · {prediction.get('checkpoint', 'checkpoint unavailable')}")
            if prediction.get("error"):
                st.error(f"Prediction unavailable: {prediction['error']}")
                continue
            predicted_class = prediction["prediction"]
            st.markdown(f"### {ICONS.get(predicted_class, '🛰️')} {predicted_class}")
            st.metric("Confidence", f"{prediction['confidence']:.1%}")
            st.markdown("**Class probabilities**")
            for class_name, probability in zip(classes, prediction["probabilities"]):
                st.markdown(f"{class_name} · {probability:.1%}")
                st.progress(probability)
