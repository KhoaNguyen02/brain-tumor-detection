import cv2
import streamlit as st
from connect import *


def main():
    # Set page configuration
    st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="wide")
    st.markdown(
        """
        <style>
        .main {
            max-width: 900px;
            margin: 0 auto;
        }
        .css-1v3fvcr { max-width: 100% !important; }
        .css-1h1c8gk { max-width: 100% !important; }
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
    uploaded_file = st.file_uploader(
        "Upload an MRI image...", type=["jpg", "jpeg", "png"])

    model = load_model()
    transform = get_transform()

    if uploaded_file is not None:
        # Process the uploaded image
        image = process_image(uploaded_file, transform)
        # Get prediction and confidence
        prediction, confidence, class_probs = predict_image(
            model, image, device)
        # Resize the image for display
        image = cv2.imread("temp/uploaded_img.jpg")
        resized_image = cv2.resize(image, (224, 224))
        # Display the uploaded image and prediction statistics
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(resized_image, caption="Uploaded Image", use_column_width=True)
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
    st.markdown("© 2024 Nguyen Anh Khoa. All Rights Reserved.", unsafe_allow_html=True)


if __name__ == "__main__":
    main()