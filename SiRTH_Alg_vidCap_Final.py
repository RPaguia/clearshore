import cv2
from picamera2 import Picamera2
import time
import numpy as np
import psutil
import subprocess
from threading import Thread
import os

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
font_size = 0.3
font_thickness = 1
line_height = 15
myColor = (0, 255, 0)
bbox_color = (0, 255, 0)
bbox_thickness = 1

# Threshold values
low_thresh = 99
high_thresh = 255

# Variables for bounding box
drawing = False
start_point = None
end_point = None
bbox = None
save_snapshot_flag = False

# Initialize metrics
cpu_usage, memory_usage, disk_usage = 0, 0, 0
bytes_sent, bytes_recv, temperature, power = 0, 0, 0, 0

# Function to resize the image to 25% of the screen width
def resize_to_quarter_screen_width(image, screen_width):
    new_width = screen_width // 4
    (h, w) = image.shape[:2]
    new_height = int(h * (new_width / w))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Function to create the "Radio" frame with spaces between frames
def create_radio_frame(cam_frame, obj_frame, thresh_frame, filtered_frame, accuracy, precision, recall, f1_score, space=10):
    cam_h, cam_w = cam_frame.shape[:2]
    obj_h, obj_w = obj_frame.shape[:2]
    thresh_h, thresh_w = thresh_frame.shape[:2]
    filtered_h, filtered_w = filtered_frame.shape[:2]

    radio_width = cam_w + thresh_w + space
    radio_height = cam_h + obj_h + space

    radio_frame = np.ones((radio_height, radio_width, 3), dtype=np.uint8) * 255

    # Position frames and add labels
    radio_frame[0:cam_h, 0:cam_w] = cam_frame
    cv2.putText(radio_frame, "Camera", (cam_w - 60, 15), font, font_size, myColor, font_thickness)

    radio_frame[cam_h + space:cam_h + space + obj_h, 0:obj_w] = obj_frame
    cv2.putText(radio_frame, "my Object", (obj_w - 60, cam_h + space + 15), font, font_size, myColor, font_thickness)
    cv2.putText(radio_frame, f'low_thresh: {low_thresh}', (10, cam_h + space + 30), font, font_size, myColor, font_thickness)
    cv2.putText(radio_frame, f'high_thresh: {high_thresh}', (10, cam_h + space + 45), font, font_size, myColor, font_thickness)

    radio_frame[0:thresh_h, cam_w + space:cam_w + space + thresh_w] = thresh_frame
    cv2.putText(radio_frame, "Threshold Mask", (cam_w + space + 240, 15), font, font_size, myColor, font_thickness)

    radio_frame[cam_h + space:cam_h + space + filtered_h, cam_w + space:cam_w + space + filtered_w] = filtered_frame
    cv2.putText(radio_frame, "Filtered Mask", (cam_w + space + 240, cam_h + space + 15), font, font_size, myColor, font_thickness)

    # Display accuracy, precision, recall, and F1-score line by line
    metrics = [
        f"Accuracy: {accuracy:.2f}%",
        f"Precision: {precision:.2f}%",
        f"Recall: {recall:.2f}%",
        f"F1-Score: {f1_score:.2f}%"
    ]
    
    for i, metric in enumerate(metrics):
        cv2.putText(radio_frame, metric, (cam_w + space + 10, cam_h + space + 60 + i * line_height), font, font_size, myColor, font_thickness)

    # Add the timestamp at the bottom of the radio frame
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text_size = cv2.getTextSize(timestamp, font, font_size, font_thickness)[0]
    text_x = (radio_width - text_size[0]) // 2  # Center the text horizontally
    text_y = cam_h + obj_h + space - 10  # Adjust the vertical position

    cv2.putText(radio_frame, timestamp, (text_x, text_y), font, font_size, myColor, font_thickness)

    return radio_frame

# Function to apply feature extraction based on surrounding pixels
def apply_surrounding_pixel_filter(mask):
    kernel = np.ones((3, 3), np.uint8)
    surrounding_count = cv2.filter2D(mask.astype(np.uint8), -1, kernel)
    filtered_mask = np.where(surrounding_count >= 6, 255, 0).astype(np.uint8)
    return filtered_mask

# Function to calculate accuracy, precision, recall, and F1-score within the bounding box
def calculate_metrics(thresh_mask, filtered_mask, bbox):
    if bbox is None:
        return 0, 0, 0, 0

    (x1, y1), (x2, y2) = bbox
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, thresh_mask.shape[1]), min(y2, thresh_mask.shape[0])

    bbox_thresh_mask = thresh_mask[y1:y2, x1:x2]
    bbox_filtered_mask = filtered_mask[y1:y2, x1:x2]

    TP = np.sum((bbox_thresh_mask == 255) & (bbox_filtered_mask == 255))
    TN = np.sum((bbox_thresh_mask == 0) & (bbox_filtered_mask == 0))
    FP = np.sum((bbox_thresh_mask == 0) & (bbox_filtered_mask == 255))
    FN = np.sum((bbox_thresh_mask == 255) & (bbox_filtered_mask == 0))

    accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100 if (TP + TN + FP + FN) > 0 else 0
    precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0
    recall = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
    f1_score = (2 * TP / (2 * TP + FP + FN)) * 100 if (2 * TP + FP + FN) > 0 else 0

    return accuracy, precision, recall, f1_score

# Mouse callback function to draw a bounding box
def draw_bbox(event, x, y, flags, param):
    global drawing, start_point, end_point, bbox, save_snapshot_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        bbox = (start_point, end_point)
        save_snapshot_flag = True

# Thread function to gather system metrics
def update_metrics():
    global cpu_usage, memory_usage, disk_usage, bytes_sent, bytes_recv, temperature, power
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        net_io = psutil.net_io_counters()
        bytes_sent, bytes_recv = net_io.bytes_sent, net_io.bytes_recv
        temperature = float(subprocess.check_output("vcgencmd measure_temp", shell=True).decode().split('=')[1].split("'")[0])
        power = float(subprocess.check_output("vcgencmd measure_volts", shell=True).decode().split('=')[1].split('V')[0])
        time.sleep(1)

# Start the thread to update metrics
Thread(target=update_metrics, daemon=True).start()

def save_snapshot(radio_frame, low_thresh, high_thresh, output_dir="/home/skySentinel/SiRTH/videoCaps"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"snapshot_{timestamp}_({low_thresh},{high_thresh}).png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, radio_frame)

cv2.namedWindow("Radio")
cv2.setMouseCallback("Radio", draw_bbox)

while True:
    tStart = time.time()
    im = picam2.capture_array()

    # Flip the image vertically to correct inversion
    #im = cv2.flip(im, 0)  # Flip vertically

    # Resize the frames
    resized_cam = resize_to_quarter_screen_width(im, dispW)
    resized_obj = resize_to_quarter_screen_width(im, dispW)
    resized_thresh = resize_to_quarter_screen_width(im, dispW)
    resized_filtered = resize_to_quarter_screen_width(im, dispW)

    # Add FPS text and resource metrics on the "Camera" frame
    metrics_start_y = pos[1] + line_height
    cv2.putText(resized_cam, f'{int(fps)} FPS', pos, font, font_size, myColor, font_thickness)

    # Display system resource metrics
    resource_metrics = [
        f'CPU: {cpu_usage:.2f}%',
        f'Memory: {memory_usage:.2f}%',
        f'Disk: {disk_usage:.2f}%',
        f'Net Sent: {bytes_sent / (1024 * 1024):.2f} MB',
        f'Net Recv: {bytes_recv / (1024 * 1024):.2f} MB',
        f'Temp: {temperature:.2f} C',
        f'Power: {power:.2f} V'
    ]

    for i, metric in enumerate(resource_metrics):
        cv2.putText(resized_cam, metric, (pos[0], metrics_start_y + i * line_height), font, font_size, myColor, font_thickness)

    # Draw the bounding box on the resized camera frame
    if bbox:
        cv2.rectangle(resized_cam, bbox[0], bbox[1], bbox_color, bbox_thickness)

    # Apply thresholding to each channel separately
    b_channel, g_channel, r_channel = cv2.split(resized_obj)
    b_mask = cv2.inRange(b_channel, low_thresh, high_thresh)
    g_mask = cv2.inRange(g_channel, low_thresh, high_thresh)
    r_mask = cv2.inRange(r_channel, low_thresh, high_thresh)

    # Combine the masks
    combined_mask = cv2.bitwise_or(b_mask, g_mask)
    combined_mask = cv2.bitwise_or(combined_mask, r_mask)

    # Apply surrounding pixel filter
    filtered_mask = apply_surrounding_pixel_filter(combined_mask)

    # Create a blank black image
    black_img = np.zeros_like(resized_obj)

    # Use the combined mask to select pixels from the original frame
    obj_frame = np.where(combined_mask[:, :, None] == 255, resized_obj, black_img)

    # Convert masks to 3-channel images for display
    combined_mask_3ch = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    filtered_mask_3ch = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)

    # Calculate accuracy, precision, recall, and F1-score within the bounding box
    accuracy, precision, recall, f1_score = calculate_metrics(combined_mask, filtered_mask, bbox)

    # Create the "Radio" frame with spaces
    radio_frame = create_radio_frame(resized_cam, obj_frame, combined_mask_3ch, filtered_mask_3ch, accuracy, precision, recall, f1_score)

    # Display the resulting frame
    cv2.imshow("Radio", radio_frame)

    # Save a snapshot if a bounding box was drawn
    if save_snapshot_flag:
        save_snapshot(radio_frame, low_thresh, high_thresh)
        save_snapshot_flag = False

    if cv2.waitKey(1) == ord('q'):
        break

    tEnd = time.time()
    loopTime = tEnd - tStart
    fps = 0.9 * fps + 0.1 * (1 / loopTime)

cv2.destroyAllWindows()
