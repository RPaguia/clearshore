import cv2
import numpy as np
import glob
import os
import pandas as pd
from datetime import datetime

# Function to load image files from subfolders within a directory
def load_images_from_folder(folder):
    images = []
    filenames = []
    for root, dirs, files in os.walk(folder):
        for file_type in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            for filename in glob.glob(os.path.join(root, file_type)):
                img = cv2.imread(filename)
                if img is not None:
                    images.append(img)
                    filenames.append(filename)
                else:
                    print(f"Failed to load {filename}")
    return images, filenames

# Function to create the Ground Truth Mask
def create_ground_truth_mask(target_img, reference_images, target_filename, target_folder):
    # Convert target image to grayscale
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize the maximum reference image
    max_reference_gray = np.zeros(target_gray.shape, dtype=np.uint8)
    
    # Centermost pixel coordinates
    center_x = target_gray.shape[1] // 2
    center_y = target_gray.shape[0] // 2

    # Determine the highest observed pixel values from reference images
    for reference_img, reference_filename in reference_images:
        resized_ref_img = cv2.resize(reference_img, (target_gray.shape[1], target_gray.shape[0]))
        reference_gray = cv2.cvtColor(resized_ref_img, cv2.COLOR_BGR2GRAY)
        max_reference_gray = np.maximum(max_reference_gray, reference_gray)
        
        # Print debug statements for the current reference image
        print(f"Target image: {target_filename}")
        print(f"Centermost pixel value in target image: {target_gray[center_y, center_x]}")
        print(f"Reference image: {reference_filename}")
        print(f"Centermost pixel value in reference image: {reference_gray[center_y, center_x]}")

    # Print debug statements for the max_reference_gray
    print(f"Centermost pixel value in max_reference_gray: {max_reference_gray[center_y, center_x]}")

    threshold_metrics_list = []
    filtered_metrics_list = []

    # Evaluate ground truth mask based on dynamic comparison for each factor
    for factor in np.arange(1.00, 5.50, 0.50):  # Increment by 50% up to 500%
        # Generate the ground truth mask for the current factor
        current_ground_truth_mask = np.where(target_gray > factor * max_reference_gray, 255, 0).astype(np.uint8)

        # Save the individual ground truth mask for the current factor
        ground_truth_mask_filename = f"Ground_Truth_{int(factor*100)}.png"
        save_mask(current_ground_truth_mask, "Ground_Truth", 0, 0, 0, 0, 0, 0, target_folder, ground_truth_mask_filename)
        
        # Generate and save threshold and filtered masks
        frameHSV = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
        hueLow, hueHigh = 0, 179
        satLow, satHigh = 0, 255
        valLow, valHigh = 186, 255
        threshold_mask = cv2.inRange(frameHSV, np.array([hueLow, satLow, valLow]), np.array([hueHigh, satHigh, valHigh]))

        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(threshold_mask // 255, -1, kernel) - (threshold_mask // 255)
        filtered_mask = np.where(neighbor_count >= 6, 255, 0).astype(np.uint8)

        # Save the threshold and filtered masks for the current factor
        threshold_mask_filename = f"Threshold_{int(factor*100)}.png"
        save_mask(threshold_mask, "Threshold", hueLow, hueHigh, satLow, satHigh, valLow, valHigh, target_folder, threshold_mask_filename)
        
        filtered_mask_filename = f"Filtered_{int(factor*100)}.png"
        save_mask(filtered_mask, "Filtered", hueLow, hueHigh, satLow, satHigh, valLow, valHigh, target_folder, filtered_mask_filename)

        # Print the result for the centermost pixel in the current ground truth mask
        print(f"Result for centermost pixel in ground truth mask at {factor*100:.0f}% threshold: {'cloud' if current_ground_truth_mask[center_y, center_x] == 255 else 'non-cloud'}")

        # Calculate metrics between the masks and the current ground truth mask
        accuracy_threshold, precision_threshold, recall_threshold, f1_score_threshold, TP_threshold, TN_threshold, FP_threshold, FN_threshold = calculate_metrics(threshold_mask, current_ground_truth_mask)
        accuracy_filtered, precision_filtered, recall_filtered, f1_score_filtered, TP_filtered, TN_filtered, FP_filtered, FN_filtered = calculate_metrics(filtered_mask, current_ground_truth_mask)

        threshold_metrics_list.append({
            "Factor": factor,
            "Filename": target_filename,
            "Total Pixels in Threshold Mask": np.size(threshold_mask),
            "Cloud Pixels in Threshold Mask": np.sum(threshold_mask == 255),
            "Non-cloud Pixels in Threshold Mask": np.size(threshold_mask) - np.sum(threshold_mask == 255),
            "TP in Threshold Mask": TP_threshold,
            "FP in Threshold Mask": FP_threshold,
            "TN in Threshold Mask": TN_threshold,
            "FN in Threshold Mask": FN_threshold,
            "Accuracy in Threshold Mask": accuracy_threshold,
            "Precision in Threshold Mask": precision_threshold,
            "Recall in Threshold Mask": recall_threshold,
            "F1-Score in Threshold Mask": f1_score_threshold
        })

        filtered_metrics_list.append({
            "Factor": factor,
            "Filename": target_filename,
            "Total Pixels in Filtered Mask": np.size(filtered_mask),
            "Cloud Pixels in Filtered Mask": np.sum(filtered_mask == 255),
            "Non-cloud Pixels in Filtered Mask": np.size(filtered_mask) - np.sum(filtered_mask == 255),
            "TP in Filtered Mask": TP_filtered,
            "FP in Filtered Mask": FP_filtered,
            "TN in Filtered Mask": TN_filtered,
            "FN in Filtered Mask": FN_filtered,
            "Accuracy in Filtered Mask": accuracy_filtered,
            "Precision in Filtered Mask": precision_filtered,
            "Recall in Filtered Mask": recall_filtered,
            "F1-Score in Filtered Mask": f1_score_filtered
        })

    return threshold_metrics_list, filtered_metrics_list

# Function for calculating accuracy between two masks and additional metrics
def calculate_metrics(detected_mask, reference_mask):
    TP = np.sum((detected_mask == 255) & (reference_mask == 255))
    TN = np.sum((detected_mask == 0) & (reference_mask == 0))
    FP = np.sum((detected_mask == 255) & (reference_mask == 0))
    FN = np.sum((detected_mask == 0) & (reference_mask == 255))

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score, TP, TN, FP, FN

# Function to draw bounding boxes around cloud regions
def draw_bounding_boxes(image, mask):
    height, width, _ = image.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area = 0
    best_bbox = None

    for i in range(1, num_labels):  # Skip the background label (0)
        x, y, w, h, area = stats[i]
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y
        component_mask = (labels == i).astype(np.uint8)
        cloud_pixel_percentage = (np.sum(component_mask) / area) * 100
        if cloud_pixel_percentage >= 50 and area > max_area:
            max_area = area
            best_bbox = (x, y, w, h)

    if best_bbox is not None:
        x, y, w, h = best_bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

# Function to generate a unique filename for the report
def generate_report_filename(output_folder, hueLow, hueHigh, satLow, satHigh, valLow, valHigh, dataset_folder, extension):
    counter = 1
    base_filename = f"({hueLow},{hueHigh})_({satLow},{satHigh})_({valLow},{valHigh})_{os.path.basename(dataset_folder)}_metrics"
    while os.path.exists(os.path.join(output_folder, f"{base_filename}_{counter}.{extension}")):
        counter += 1
    return os.path.join(output_folder, f"{base_filename}_{counter}.{extension}")

# Function to save mask with specific filename format
def save_mask(mask, mask_type, hueLow, hueHigh, satLow, satHigh, valLow, valHigh, dataset_folder, original_filename):
    mask_folder = f"/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/masks"
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    mask_filename = f"{mask_type}Mask_({hueLow},{hueHigh})({satLow},{satHigh})({valLow},{valHigh})_{os.path.basename(dataset_folder)}_{os.path.basename(original_filename)}"
    cv2.imwrite(os.path.join(mask_folder, mask_filename), mask)

# Mapping of target folders to reference folders
folder_mapping = {
    "HG_L8_clear": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/HG_L8_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_L8_clear"],
    "HG_L8_sunny": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/HG_L8_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_L8_clear"],
    "HG_L8_cloudy": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/HG_L8_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_L8_clear"],
    "HG_L8_overcast": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/HG_L8_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_L8_clear"],
    "HG_S2_clear": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/HG_S2_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_S2_clear"],
    "HG_S2_sunny": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/HG_S2_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_S2_clear"],
    "HG_S2_cloudy": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/HG_S2_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_S2_clear"],
    "HG_S2_overcast": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/HG_S2_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_S2_clear"],
    "TA_L8_clear": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TA_L8_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_L8_clear"],
    "TA_L8_sunny": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TA_L8_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_L8_clear"],
    "TA_L8_cloudy": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TA_L8_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_L8_clear"],
    "TA_L8_overcast": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TA_L8_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_L8_clear"],
    "TA_S2_clear": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TA_S2_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_S2_clear"],
    "TA_S2_sunny": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TA_S2_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_S2_clear"],
    "TA_S2_cloudy": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TA_S2_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_S2_clear"],
    "TA_S2_overcast": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TA_S2_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_S2_clear"],
    "TU_L8_clear": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TU_L8_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_L8_clear"],
    "TU_L8_sunny": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TU_L8_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_L8_clear"],
    "TU_L8_cloudy": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TU_L8_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_L8_clear"],
    "TU_L8_overcast": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TU_L8_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_L8_clear"],
    "TU_S2_clear": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TU_S2_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_S2_clear"],
    "TU_S2_sunny": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TU_S2_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_S2_clear"],
    "TU_S2_cloudy": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TU_S2_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_S2_clear"],
    "TU_S2_overcast": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/TU_S2_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_S2_clear"],
    "MJ_L8_clear": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/MJ_L8_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_L8_clear"],
    "MJ_L8_sunny": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/MJ_L8_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_L8_clear"],
    "MJ_L8_cloudy": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/MJ_L8_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_L8_clear"],
    "MJ_L8_overcast": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/MJ_L8_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_L8_clear"],
    "MJ_S2_clear": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/MJ_S2_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_S2_clear"],
    "MJ_S2_sunny": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/MJ_S2_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_S2_clear"],
    "MJ_S2_cloudy": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/MJ_S2_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_S2_clear"],
    "MJ_S2_overcast": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/sky/MJ_S2_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_S2_clear"],
    "HG_L8_summer": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/HG_L8_summer", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_L8_summer"],
    "HG_L8_fall": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/HG_L8_fall", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_L8_fall"],
    "HG_L8_winter": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/HG_L8_winter", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_L8_winter"],
    "HG_L8_spring": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/HG_L8_spring", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_L8_spring"],
    "HG_S2_summer": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/HG_S2_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_S2_summer"],
    "HG_S2_fall": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/HG_S2_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_S2_fall"],
    "HG_S2_winter": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/HG_S2_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_S2_winter"],
    "HG_S2_spring": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/HG_S2_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Hauraki_Gulf/HG_S2_spring"],
    "TA_L8_summer": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TA_L8_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_L8_summer"],
    "TA_L8_fall": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TA_L8_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_L8_fall"],
    "TA_L8_winter": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TA_L8_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_L8_winter"],
    "TA_L8_spring": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TA_L8_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_L8_spring"],
    "TA_S2_summer": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TA_S2_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_S2_summer"],
    "TA_S2_fall": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TA_S2_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_S2_fall"],
    "TA_S2_winter": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TA_S2_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_S2_winter"],
    "TA_S2_spring": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TA_S2_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tahoe/TA_S2_spring"],
    "TU_L8_summer": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TU_L8_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_L8_summer"],
    "TU_L8_fall": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TU_L8_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_L8_fall"],
    "TU_L8_winter": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TU_L8_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_L8_winter"],
    "TU_L8_spring": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TU_L8_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_L8_spring"],
    "TU_S2_summer": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TU_S2_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_S2_summer"],
    "TU_S2_fall": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TU_S2_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_S2_fall"],
    "TU_S2_winter": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TU_S2_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_S2_winter"],
    "TU_S2_spring": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/TU_S2_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Lake_Tuggeranong/TU_S2_spring"],
    "MJ_L8_summer": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/MJ_L8_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_L8_summer"],
    "MJ_L8_fall": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/MJ_L8_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_L8_fall"],
    "MJ_L8_winter": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/MJ_L8_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_L8_winter"],
    "MJ_L8_spring": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/MJ_L8_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_L8_spring"],
    "MJ_S2_summer": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/MJ_S2_clear", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_S2_summer"],
    "MJ_S2_fall": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/MJ_S2_sunny", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_S2_fall"],
    "MJ_S2_winter": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/MJ_S2_cloudy", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_S2_winter"],
    "MJ_S2_spring": ["/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/images/season/MJ_S2_overcast", "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/cloudFreeImages/Mount_John/MJ_S2_spring"]
}

# Iterate through each target and reference folder pair in the mapping
for key, paths in folder_mapping.items():
    target_folder = paths[0]
    reference_folder = paths[1]

    # Load target images from the specified folder
    target_images, target_filenames = load_images_from_folder(target_folder)

    # Load reference images from the specified folder
    reference_images, reference_filenames = load_images_from_folder(reference_folder)

    # Check if images are loaded
    if len(target_images) == 0 or len(reference_images) == 0:
        print(f"No images found in the specified folder for {key}.")
        continue

    # Define the output folder
    output_folder = "/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/100_groundTruth/output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize lists to store metrics for the Excel report
    threshold_metrics_list = []
    filtered_metrics_list = []

    # Loop through all target images to process and save masks
    for idx, target_img in enumerate(target_images):
        start_time = datetime.now()
        filename = target_filenames[idx]

        # Create Ground Truth Mask
        threshold_metrics, filtered_metrics = create_ground_truth_mask(target_img, zip(reference_images, reference_filenames), filename, target_folder)
        
        # Extend the metrics lists with the increment metrics
        threshold_metrics_list.extend(threshold_metrics)
        filtered_metrics_list.extend(filtered_metrics)

    # Save the metrics to an Excel file with a unique name
    excel_filename = generate_report_filename(output_folder, 0, 179, 0, 255, 186, 255, target_folder, 'xlsx')
    with pd.ExcelWriter(excel_filename) as writer:
        pd.DataFrame(threshold_metrics_list).to_excel(writer, sheet_name='Threshold Mask Metrics', index=False)
        pd.DataFrame(filtered_metrics_list).to_excel(writer, sheet_name='Filtered Mask Metrics', index=False)

    # Save the metrics to a CSV file with a unique name
    csv_filename = generate_report_filename(output_folder, 0, 179, 0, 255, 186, 255, target_folder, 'csv')
    pd.DataFrame(threshold_metrics_list).to_csv(f"{csv_filename}_threshold.csv", index=False)
    pd.DataFrame(filtered_metrics_list).to_csv(f"{csv_filename}_filtered.csv", index=False)

    print(f"Reports saved to: {excel_filename} and {csv_filename}_threshold.csv, {csv_filename}_filtered.csv")

