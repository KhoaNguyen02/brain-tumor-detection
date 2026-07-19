import os

import cv2

from config import ClassifierConfig, SegmentationConfig, device
from models import AttentionUNet, ConvNext, draw_segmentation, predict
from preprocessing.dataset import CLASS_NAMES
from training import load_model


def get_models():
    _, classifier = load_model('Classifier', model_class=lambda: ConvNext(num_classes=len(CLASS_NAMES)), get_model=True, device=device)
    classifier.eval()
    classifier_config = ClassifierConfig(classifier)

    _, segmenter = load_model('Segmenter', model_class=AttentionUNet, get_model=True, device=device)
    segmenter.eval()
    segmenter_config = SegmentationConfig(segmenter)

    return classifier, segmenter, classifier_config, segmenter_config


def process_image(uploaded_file):
    os.makedirs('temp', exist_ok=True)
    image_path = 'temp/uploaded_img.jpg'
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return image_path


def run_pipeline(classifier, segmenter, classifier_config, segmenter_config, image_path, device, class_names):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))

    result = predict(classifier, segmenter, image, classifier_config, segmenter_config, class_names, device)
    annotated = draw_segmentation(image, result, class_names)

    return annotated, result
