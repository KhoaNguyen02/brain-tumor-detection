import cv2
import streamlit as st
from models import *
from preprocessing import *
from torchvision import transforms

MODEL = "pretrained/model.pth"

def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        #                      0.229, 0.224, 0.225])
    ])

    # Streamlit app
    st.title("Brain Tumor Classifier")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp/uploaded_img.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and transform the image
        image = load_single_img("temp/uploaded_img.jpg",transform)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CvT(num_classes=4).to(device)
        model.load_state_dict(torch.load(MODEL, map_location=device))

        # Get prediction and confidence
        prediction, confidence = predict(model, image, device)

        # Display the results
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Generate and display the heatmap
        input_img = next(iter(image))[0].unsqueeze(0).to(device)

        # Clear hooks if needed
        for layer in model.modules():
            clear_hooks(layer)

        register_hooks(model)

        # Generate the CAM
        heatmap = generate_cam(model, input_img)  # Assuming single image

        # Apply the heatmap on the image
        img = input_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.uint8(255 * img)
        superimposed_img = apply_heatmap(img, heatmap)
        heatmap_path = 'temp/heatmap.jpg'
        cv2.imwrite(heatmap_path, superimposed_img)

        # Display the uploaded image and the heatmap side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image("temp/uploaded_img.jpg",
                    caption="Uploaded Image", use_column_width=True)

        with col2:
            st.image(heatmap_path, caption="Activation Map", use_column_width=True)


if __name__ == "__main__":
    main()