import cv2
from picamera2 import Picamera2
import time
import numpy as np
import random

# Initialize the camera
picam2 = Picamera2()
dispW = 1280
dispH = 720
picam2.preview_configuration.main.size = (dispW, dispH)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 30
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

fps = 0
pos = (20, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
height = 0.8
weight = 2
myColor = (0, 0, 255)
height1 = 0.3
myColor1 = (0, 255, 0)
weight1 = 1

# Threshold values
low_thresh = 29
high_thresh = 74

# Function to resize the image to 25% of the screen width
def resize_to_quarter_screen_width(image, screen_width):
    new_width = screen_width // 4
    (h, w) = image.shape[:2]
    new_height = int(h * (new_width / w))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Function to create the "Radio" frame with spaces between frames
def create_radio_frame(cam_frame, obj_frame, thresh_frame, filtered_frame, accuracy, space=10):
    cam_h, cam_w = cam_frame.shape[:2]
    obj_h, obj_w = obj_frame.shape[:2]
    thresh_h, thresh_w = thresh_frame.shape[:2]
    filtered_h, filtered_w = filtered_frame.shape[:2]

    radio_width = cam_w + thresh_w + space
    radio_height = cam_h + obj_h + space

    radio_frame = np.ones((radio_height, radio_width, 3), dtype=np.uint8) * 255

    # Position frames and add labels
    radio_frame[0:cam_h, 0:cam_w] = cam_frame
    cv2.putText(radio_frame, "Camera", (cam_w - 60, 15), font, height1, myColor1, weight1)

    radio_frame[cam_h + space:cam_h + space + obj_h, 0:obj_w] = obj_frame
    cv2.putText(radio_frame, "my Object", (obj_w - 60, cam_h + space + 15), font, height1, myColor1, weight1)

    radio_frame[0:thresh_h, cam_w + space:cam_w + space + thresh_w] = thresh_frame
    cv2.putText(radio_frame, "Threshold Mask", (cam_w + space + 240, 15), font, height1, myColor1, weight1)

    radio_frame[cam_h + space:cam_h + space + filtered_h, cam_w + space:cam_w + space + filtered_w] = filtered_frame
    cv2.putText(radio_frame, "Filtered Mask", (cam_w + space + 240, cam_h + space + 15), font, height1, myColor1, weight1)

    # Display accuracy within the Filtered Mask frame
    accuracy_text = f"Accuracy: {accuracy:.2f}%"
    cv2.putText(radio_frame, accuracy_text, (cam_w + space + 20, cam_h + space + 20), font, height, myColor, weight)

    return radio_frame

# Function to apply feature extraction based on surrounding pixels
def apply_surrounding_pixel_filter(mask):
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel of ones
    surrounding_count = cv2.filter2D((mask == 255).astype(np.uint8), -1, kernel)  # Count surrounding pixels that are 255

    # Apply threshold: a pixel in the filtered mask is set to 255 if 6 or more surrounding pixels are 255 in the threshold mask
    filtered_mask = np.where(surrounding_count >= 6, 255, 0).astype(np.uint8)  # Apply threshold

    # Debugging: Count how many pixels were changed from cloud to non-cloud
    changes = np.sum((mask == 255) & (filtered_mask == 0))
    print(f"Pixels changed from cloud to non-cloud: {changes}")

    return filtered_mask

# Function to calculate metrics
def calculate_metrics(TP, FP, TN, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

# Function to calculate accuracy and metrics based on the given criteria within the bounding box
def calculate_accuracy_and_metrics(mask1, mask2, bbox=None):
    if bbox:
        y1, x1, y2, x2 = bbox
        mask1 = mask1[y1:y2, x1:x2]
        mask2 = mask2[y1:y2, x1:x2]

    TP = np.sum((mask1 == 255) & (mask2 == 255))
    TN = np.sum((mask1 == 0) & (mask2 == 0))
    FP = np.sum((mask1 == 255) & (mask2 == 0))
    FN = np.sum((mask1 == 0) & (mask2 == 255))

    if (TP + TN + FP + FN) > 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    else:
        accuracy = 0

    precision, recall, f1_score = calculate_metrics(TP, FP, TN, FN)

    return accuracy, TP, FP, TN, FN, precision, recall, f1_score

# Function to generate random bounding box within thresholds
def generate_random_bounding_box(mask, low_thresh, high_thresh, required_density=0.5):
    h, w = mask.shape
    attempts = 0
    while attempts < 100:  # Try up to 100 times to find a suitable bounding box
        box_width = random.randint(50, w // 2)
        box_height = random.randint(50, h // 2)
        x1 = random.randint(0, w - box_width)
        y1 = random.randint(0, h - box_height)
        x2 = x1 + box_width
        y2 = y1 + box_height

        # Calculate the percentage of cloud pixels within the bounding box
        bbox_area = mask[y1:y2, x1:x2]
        cloud_pixels = np.sum(bbox_area == 255)
        total_pixels = bbox_area.size

        if (cloud_pixels / total_pixels) >= required_density:  # Check if the density of cloud pixels is at least 50%
            return (y1, x1, y2, x2)  # Return the bounding box coordinates

        attempts += 1

    return None  # Return None if no suitable box found after 100 attempts

# Main loop
snapshot_path = '/media/skyCloud/MYCRUZERUSB/cdm-mlb-cls/radioThresh/SiRTHAlgorithm/vidCaptures'
for i in range(5):
    tStart = time.time()
    im = picam2.capture_array()

    # Rotate the frame to ensure it is in landscape orientation
    if im.shape[0] > im.shape[1]:  # Check if the frame is in portrait mode
        im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Resize the frames
    resized_cam = resize_to_quarter_screen_width(im, dispW)
    resized_obj = resize_to_quarter_screen_width(im, dispW)

    # Apply thresholding directly to generate mask1
    b_channel, g_channel, r_channel = cv2.split(resized_obj)
    b_mask = cv2.inRange(b_channel, low_thresh, high_thresh)
    g_mask = cv2.inRange(g_channel, low_thresh, high_thresh)
    r_mask = cv2.inRange(r_channel, low_thresh, high_thresh)
    mask1 = cv2.bitwise_or(b_mask, g_mask)
    mask1 = cv2.bitwise_or(mask1, r_mask)

    # Print debug statements for mask1
    total_pixels_mask1 = mask1.size
    cloud_pixels_mask1 = np.sum(mask1 == 255)
    non_cloud_pixels_mask1 = np.sum(mask1 == 0)
    print(f" ")
    print(f"Total No. of pixels in Threshold Mask: {total_pixels_mask1}")
    print(f"No. of cloud pixels in Threshold Mask: {cloud_pixels_mask1}")
    print(f"No. of non-cloud pixels in Threshold Mask: {non_cloud_pixels_mask1}")

    # Apply surrounding pixel filter to generate mask2
    mask2 = apply_surrounding_pixel_filter(mask1)

    # Print debug statements for mask2
    total_pixels_mask2 = mask2.size
    cloud_pixels_mask2 = np.sum(mask2 == 255)
    non_cloud_pixels_mask2 = np.sum(mask2 == 0)
    print(f" ")
    print(f"Total No. of pixels in Filtered Mask: {total_pixels_mask2}")
    print(f"No. of cloud pixels in Filtered Mask: {cloud_pixels_mask2}")
    print(f"No. of non-cloud pixels in Filtered Mask: {non_cloud_pixels_mask2}")

    # Calculate accuracy and metrics for the entire image
    accuracy, TP, FP, TN, FN, precision, recall, f1_score = calculate_accuracy_and_metrics(mask1, mask2)
    print(f" ")
    print(f"Referenced to the Filtered Image - TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Referenced to the Filtered Image - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}")

    # Generate a random bounding box that contains at least 50% cloud pixels
    bbox = generate_random_bounding_box(mask1, low_thresh, high_thresh)

    if bbox:
        y1, x1, y2, x2 = bbox
        cv2.rectangle(resized_cam, (x1, y1), (x2, y2), myColor, 2)

        # Print additional debug statements for bounding box
        bbox_total_pixels_mask1 = (y2 - y1) * (x2 - x1)
        bbox_cloud_pixels_mask1 = np.sum(mask1[y1:y2, x1:x2] == 255)
        bbox_non_cloud_pixels_mask1 = np.sum(mask1[y1:y2, x1:x2] == 0)
        bbox_changes = np.sum((mask1[y1:y2, x1:x2] == 255) & (mask2[y1:y2, x1:x2] == 0))

        print(f" ")
        print(f"Total No. of pixels within the Bounding Box in Threshold Mask: {bbox_total_pixels_mask1}")
        print(f"No. of cloud pixels within the Bounding Box in Threshold Mask: {bbox_cloud_pixels_mask1}")
        print(f"No. of non-cloud pixels within the Bounding Box in Threshold Mask: {bbox_non_cloud_pixels_mask1}")
        print(f"Pixels changed from cloud to non-cloud within the Bounding Box: {bbox_changes}")

        bbox_total_pixels_mask2 = (y2 - y1) * (x2 - x1)
        bbox_cloud_pixels_mask2 = np.sum(mask2[y1:y2, x1:x2] == 255)
        bbox_non_cloud_pixels_mask2 = np.sum(mask2[y1:y2, x1:x2] == 0)

        print(f" ")
        print(f"Total No. of pixels within the Bounding Box in Filtered Mask: {bbox_total_pixels_mask2}")
        print(f"No. of cloud pixels within the Bounding Box in Filtered Mask: {bbox_cloud_pixels_mask2}")
        print(f"No. of non-cloud pixels within the Bounding Box in Filtered Mask: {bbox_non_cloud_pixels_mask2}")
        
                # Calculate accuracy and metrics within the bounding box
        bbox_accuracy, bbox_TP, bbox_FP, bbox_TN, bbox_FN, bbox_precision, bbox_recall, bbox_f1_score = calculate_accuracy_and_metrics(mask1, mask2, bbox)
        print(f" ")
        print(f"Referenced to the Bounding Box - TP: {bbox_TP}, FP: {bbox_FP}, TN: {bbox_TN}, FN: {bbox_FN}")
        print(f"Referenced to the Bounding Box - Accuracy: {accuracy:.2f}, Precision: {bbox_precision:.2f}, Recall: {bbox_recall:.2f}, F1-Score: {bbox_f1_score:.2f}")

    else:
        accuracy = 0
        print("No suitable bounding box found.")

    # Convert masks to 3-channel images for display
    mask1_3ch = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
    mask2_3ch = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)

    # Add FPS text to the Camera frame
    cv2.putText(resized_cam, str(int(fps)) + ' FPS', pos, font, height, myColor, weight)

    # Create the "Radio" frame with spaces
    radio_frame = create_radio_frame(resized_cam, resized_obj, mask1_3ch, mask2_3ch, accuracy)

    # Display the resulting frame
    cv2.imshow("Radio", radio_frame)

    # Save snapshot of Radio frame
    cv2.imwrite(f'{snapshot_path}/snapshot_{i + 1}.jpg', radio_frame)

    if cv2.waitKey(1) == ord('q'):
        break

    tEnd = time.time()
    loopTime = tEnd - tStart
    fps = 0.9 * fps + 0.1 * (1 / loopTime)

    # Wait 3 seconds
    time.sleep(3)

cv2.destroyAllWindows()
