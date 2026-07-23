import json
import os

import streamlit as st

from config import CLASS_NAMES, device
from connect import get_models, process_image, run_pipeline

ACCENT = "#4361EE"
DANGER = "#E63946"
SUCCESS = "#2A9D8F"
WARNING = "#F4A261"


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }

        .hero {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
            border-radius: 18px;
            padding: 2.2rem 2.6rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.25);
        }
        .hero h1 { margin: 0; font-size: 2.05rem; font-weight: 800; color: white; }
        .hero p { margin: 0.6rem 0 0 0; font-size: 1.02rem; color: rgba(255,255,255,0.92); }
        .hero .pipeline-tags { margin-top: 1rem; }
        .tag {
            display: inline-block;
            background: rgba(255,255,255,0.16);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }

        .result-badge {
            display: inline-block;
            padding: 0.4rem 1.1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.15rem;
            margin-bottom: 1rem;
        }
        .badge-danger { background: rgba(230, 57, 70, 0.14); color: #E63946; }
        .badge-success { background: rgba(42, 157, 143, 0.14); color: #2A9D8F; }
        .badge-warning { background: rgba(244, 162, 97, 0.16); color: #F4A261; }

        .prob-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.6rem; }
        .prob-label { min-width: 108px; font-size: 0.87rem; font-weight: 600; }
        .prob-track { flex: 1; height: 10px; background: rgba(128,128,128,0.18); border-radius: 999px; overflow: hidden; }
        .prob-fill { height: 100%; border-radius: 999px; }
        .prob-pct { min-width: 46px; font-size: 0.85rem; text-align: right; opacity: 0.7; }

        .empty-state {
            text-align: center;
            padding: 2.5rem 1rem;
            opacity: 0.65;
        }
        .empty-state .icon { font-size: 2.4rem; margin-bottom: 0.5rem; }

        footer { visibility: hidden; }
        .app-footer { text-align: center; opacity: 0.5; font-size: 0.82rem; margin-top: 2.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>🧠 Brain Tumor Detection</h1>
            <p>Upload a brain MRI slice to detect tumors and segment them when present.</p>
            <div class="pipeline-tags">
                <span class="tag">ConvNeXt-Tiny Classifier</span>
                <span class="tag">Attention U-Net Segmenter</span>
                <span class="tag">BRISC2025</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_test_metrics(model_name):
    path = f'pretrained/{model_name}/{model_name}_history.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)['test_metrics']


def render_sidebar():
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "A medical imaging application that classifies brain MRI slices and segments tumors if present."
        )
        st.markdown("**Classifier**: ConvNeXt-Tiny")
        st.markdown("**Segmenter**: Attention U-Net")
        st.markdown("**Dataset**: [BRISC2025](https://arxiv.org/abs/2506.14318)")

        clf_metrics = load_test_metrics('Classifier')
        seg_metrics = load_test_metrics('Segmenter')
        if clf_metrics or seg_metrics:
            st.divider()
            st.markdown("**Reported test performance**")
            c1, c2 = st.columns(2)
            if clf_metrics:
                c1.metric("Classification acc.", f"{clf_metrics['accuracy']:.2%}")
            if seg_metrics:
                c2.metric("Segmentation acc.", f"{seg_metrics['dice'] * 100:.2f}%")

        st.divider()
        st.caption("© 2026 Nguyen Anh Khoa. All Rights Reserved.")


def render_probability_bars(class_probs, predicted_idx, ambiguous=False, unknown_prob=0.0):
    entries = list(zip(CLASS_NAMES, class_probs))
    if ambiguous:
        entries.append(('unknown other type', unknown_prob))

    order = sorted(range(len(entries)), key=lambda i: entries[i][1], reverse=True)
    rows = []
    for i in order:
        label, prob = entries[i]
        if ambiguous and label == 'unknown other type':
            color, opacity = WARNING, 1.0
        elif i == predicted_idx:
            color, opacity = DANGER, 1.0
        else:
            color, opacity = ACCENT, 0.55
        rows.append(f"""
        <div class="prob-row">
            <div class="prob-label">{label}</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{prob * 100:.1f}%;
                    background:{color}; opacity:{opacity};"></div>
            </div>
            <div class="prob-pct">{prob * 100:.1f}%</div>
        </div>
        """)
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_results(annotated, result):
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container(border=True):
            st.image(annotated, caption="Segmentation Result", width=512)
    with col2:
        with st.container(border=True):
            st.markdown("##### 🔍 Detection Details")
            if result['box'] is None:
                st.markdown(
                    '<span class="result-badge badge-success">✅ Healthy</span>',
                    unsafe_allow_html=True)
            elif result['class_idx'] is None:
                st.markdown(
                    '<span class="result-badge badge-warning">❓ Unknown Other Type</span>',
                    unsafe_allow_html=True)
            else:
                name = CLASS_NAMES[result['class_idx']]
                st.markdown(
                    f'<span class="result-badge badge-danger">⚠️ {name.title()}</span>',
                    unsafe_allow_html=True)
                st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

            st.markdown("**Probability Breakdown**")
            render_probability_bars(
                result['class_probs'], result['class_idx'],
                ambiguous=result['ambiguous'], unknown_prob=result['unknown_prob'])


def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")
    inject_css()
    render_hero()
    render_sidebar()

    st.warning("⚠️ For demonstration purposes only. Do not rely on the predictions for medical diagnosis.")

    if 'classifier' not in st.session_state:
        with st.spinner("Loading models..."):
            (st.session_state.classifier, st.session_state.segmenter,
                st.session_state.classifier_config, st.session_state.segmenter_config) = get_models()

    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload an fMRI/MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = process_image(uploaded_file)
        with st.spinner("Running inference..."):
            annotated, result = run_pipeline(
                st.session_state.classifier, st.session_state.segmenter,
                st.session_state.classifier_config, st.session_state.segmenter_config,
                image_path, device, CLASS_NAMES)
        render_results(annotated, result)
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">📤</div>
                Upload an MRI scan above to see the classification and segmentation results.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
