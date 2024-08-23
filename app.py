import cv2
import streamlit as st
from connect import *


def main():
    # Set page configuration
    st.set_page_config(page_title="Brain Tumor Classifier",
                       page_icon="🧠", layout="wide")
    st.markdown(
        """
        <style>
        .main {
            max-width: 900px;
            margin: 0 auto;
        }
        .css-1v3fvcr { max-width: 100% !important; }
        .css-1h1c8gk { max-width: 100% !important; }
        .model-selector {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit app layout
    st.title("🧠 Brain Tumor Classifier")
    st.markdown(
        "This app classifies brain tumors into four different categories including no tumor, glioma, meningioma, and pituitary tumors."
    )
    st.warning(
        "⚠️ For demonstration purposes only. Do not rely on the predictions for medical diagnosis."
    )

    # Model selection
    if 'model' not in st.session_state:
        st.session_state.model = None

    model_option = st.selectbox(
        "Select a Model",
        ["None", "Auto", "CNN", "ResNet", "DenseNet"],
        key="model_selector"
    )

    if model_option != "None":
        if st.session_state.model is None or st.session_state.model_name != model_option:
            st.session_state.model_name = model_option
            st.session_state.model, st.session_state.config = get_model(
                model_option)
            st.success(f"Model `{model_option}` loaded successfully!")

    if st.session_state.model is None:
        st.info("Please select a model from the dropdown to enable image upload.")
    else:
        uploaded_file = st.file_uploader(
            "Upload an fMRI/MRI image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image_path = process_image(uploaded_file)
            # Get prediction and confidence
            prediction, confidence, class_probs = predict_image(
                st.session_state.model, st.session_state.config, image_path, device)
            # Resize the image for display
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (224, 224))
            # Display the uploaded image and prediction statistics
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(resized_image, caption="Uploaded Image",
                        use_column_width=True)
            with col2:
                st.subheader("🔍 Prediction Details")
                st.markdown("**Predicted Condition:**")
                if prediction == "Normal":
                    st.markdown(
                        f"<h3 style='color:#1db954;'>{prediction}</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<h3 style='color:#ff4b4b;'>{prediction}</h3>", unsafe_allow_html=True)
                st.markdown("**Confidence Level:**")
                st.progress(int(confidence))
                st.markdown(f"**Confidence: {confidence:.3f}%**")
                st.markdown("### Probability Breakdown")
                for condition, prob in class_probs.items():
                    st.markdown(f"- **{condition}:** {prob:.3f}%")

    # Copyright
    st.markdown("---")
    st.markdown("© 2024 Nguyen Anh Khoa. All Rights Reserved.",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
