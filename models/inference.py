import cv2
import numpy as np
import torch


def largest_component_bbox(mask, min_area_fraction=0.001):
    h, w = mask.shape
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas))
    if areas[best] < min_area_fraction * mask.size:
        return None

    x, y, bw, bh, _ = stats[best + 1]
    return (x / w, y / h, (x + bw) / w, (y + bh) / h)


@torch.no_grad()
def predict(classifier, segmenter, image, classifier_config, segmenter_config, class_names, device):
    classifier.eval()
    clf_tensor = classifier_config.test_transform(image=image)['image'].unsqueeze(0).to(device)
    class_logits = classifier(clf_tensor)

    class_probs = torch.softmax(class_logits, dim=-1)[0].cpu().numpy()
    class_idx = int(class_probs.argmax())
    confidence = float(class_probs[class_idx])

    result = {
        'class_idx': class_idx,
        'confidence': confidence,
        'class_probs': class_probs,
        'mask': None,
        'box': None,
    }

    if class_names[class_idx] == 'no_tumor':
        return result

    segmenter.eval()
    seg_tensor = segmenter_config.test_transform(image=image)['image'].unsqueeze(0).to(device)
    seg_logits = segmenter(seg_tensor)

    mask_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    mask = (mask_prob > segmenter_config.seg_threshold).astype(np.uint8)
    box = largest_component_bbox(mask, min_area_fraction=segmenter_config.min_tumor_area_fraction)

    result['mask'] = mask
    result['box'] = box
    return result


def draw_segmentation(image, result, class_names):
    if result['box'] is None:
        return image.copy()

    h, w = image.shape[:2]
    mask_resized = cv2.resize(result['mask'], (w, h), interpolation=cv2.INTER_NEAREST)
    color = (255, 71, 87)

    overlay = image.copy()
    overlay[mask_resized > 0] = color
    blended = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

    x1n, y1n, x2n, y2n = result['box']
    x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
    cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)

    return blended
