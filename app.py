import streamlit as st

from config import CLASS_NAMES, device
from connect import get_models, process_image, run_pipeline


def main():
    st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠", layout="wide")
    st.markdown(
        """
        <style>
        .main { max-width: 900px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🧠 Brain Tumor Segmentation & Classification")
    st.markdown(
        "This app classifies a brain MRI scan as "
        f"**{'**, **'.join(CLASS_NAMES)}**, then segments the tumor pixel-by-pixel "
        "if one is present."
    )
    st.warning(
        "⚠️ For demonstration purposes only. Do not rely on the predictions for medical diagnosis."
    )

    if 'classifier' not in st.session_state:
        with st.spinner("Loading models..."):
            (st.session_state.classifier, st.session_state.segmenter,
                st.session_state.classifier_config, st.session_state.segmenter_config) = get_models()

    uploaded_file = st.file_uploader(
        "Upload an fMRI/MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = process_image(uploaded_file)
        annotated, result = run_pipeline(
            st.session_state.classifier, st.session_state.segmenter,
            st.session_state.classifier_config, st.session_state.segmenter_config,
            image_path, device, CLASS_NAMES)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(annotated, caption="Segmentation Result", use_column_width=True)
        with col2:
            st.subheader("🔍 Detection Details")
            if result['box'] is None:
                st.markdown(
                    "<h3 style='color:#1db954;'>No Tumor Detected</h3>", unsafe_allow_html=True)
            else:
                name = CLASS_NAMES[result['class_idx']]
                st.markdown(f"<h3 style='color:#ff4b4b;'>{name}</h3>", unsafe_allow_html=True)
                st.progress(result['confidence'])
                st.markdown(f"**Confidence:** {result['confidence'] * 100:.1f}%")
                st.markdown("### Probability Breakdown")
                for name, prob in zip(CLASS_NAMES, result['class_probs']):
                    st.markdown(f"- **{name}:** {prob * 100:.1f}%")

    # Copyright
    st.markdown("---")
    st.markdown("© 2026 Nguyen Anh Khoa. All Rights Reserved.",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
