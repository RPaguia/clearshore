import argparse
import sys
import time
import cv2
import mediapipe as mp
import psutil
import subprocess
import os
import pandas as pd
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
from picamera2 import Picamera2

# Global variables to calculate FPS and Confidence Level
COUNTER = 0
START_TIME = time.time()
confidence_level = 0.0
FPS = 0  # Initialize FPS globally to avoid UnboundLocalError

# Folder to save snapshots and reports
save_folder = '/home/skySentinel/tflite-custom-object-bookworm-main/snapshots_cloudDet_bbox'
os.makedirs(save_folder, exist_ok=True)

# List to store metrics for the final report
all_metrics = []

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def get_system_metrics():
    """Get system metrics using psutil and vcgencmd."""
    # CPU Usage
    cpu_usage = psutil.cpu_percent()

    # Memory Usage
    memory_usage = psutil.virtual_memory().percent

    # Disk Usage
    disk_usage = psutil.disk_usage('/').percent

    # Network Usage (MB)
    net_io = psutil.net_io_counters()
    net_sent = net_io.bytes_sent / (1024 * 1024)  # Convert to MB
    net_recv = net_io.bytes_recv / (1024 * 1024)  # Convert to MB

    # Temperature (Celsius)
    temp_output = subprocess.check_output("vcgencmd measure_temp", shell=True).decode()
    temperature = float(temp_output.split('=')[1].split("'")[0])

    # Power (Voltage)
    power_output = subprocess.check_output("vcgencmd measure_volts", shell=True).decode()
    power = float(power_output.split('=')[1].split('V')[0])

    return cpu_usage, memory_usage, disk_usage, net_sent, net_recv, temperature, power

def save_snapshot(image, metrics, folder, prefix='snapshot'):
    """Save the snapshot image and metrics report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(folder, f"{prefix}_{timestamp}.png")

    # Save the image
    cv2.imwrite(image_filename, image)
    print(f"Snapshot saved to {image_filename}")

    # Append filename to metrics
    metrics['Filename'] = f"{prefix}_{timestamp}"

    # Append metrics to all_metrics list
    all_metrics.append(metrics)

def save_final_report(folder, filename='final_report'):
    """Save the final report as CSV and Excel files."""
    df = pd.DataFrame(all_metrics)
    csv_filename = os.path.join(folder, f"{filename}.csv")
    excel_filename = os.path.join(folder, f"{filename}.xlsx")

    # Save the final report as CSV
    df.to_csv(csv_filename, index=False)

    # Save the final report as Excel using openpyxl engine
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    print(f"Final report saved to {csv_filename} and {excel_filename}")

def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera."""
  
    global FPS  # Use global FPS to avoid UnboundLocalError

    # Visualization parameters
    row_size = 25  # pixels, reduced for smaller font size
    right_margin = width - 500  # Starting point for text from the right side
    metrics_color = (139, 0, 0)  # Dark blue color for system metrics
    font_size = 0.4  # Reduced font size
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None
    detection_result_list = []

    # Define the colors for bounding boxes and labels
    bright_green = (0, 255, 0)  # Bright green for clouds
    bright_blue = (255, 255, 0)  # Bright blue for water bodies
    default_color = (255, 255, 255)  # Default white color for unrecognized categories

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, confidence_level

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = int(fps_avg_frame_count / (time.time() - START_TIME))  # Use int for whole number
            START_TIME = time.time()

        # Calculate confidence level as the average score of detected objects
        if result.detections:
            confidence_level = sum(detection.categories[0].score for detection in result.detections) / len(result.detections)

        detection_result_list.append(result)
        COUNTER += 1

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           max_results=max_results, score_threshold=score_threshold,
                                           result_callback=save_result)
    detector = vision.ObjectDetector.create_from_options(options)

    def mouse_callback(event, x, y, flags, param):
        """Mouse callback function to save a snapshot on click."""
        if event == cv2.EVENT_LBUTTONDOWN:
            save_snapshot(current_frame, metrics_dict, save_folder)

    # Set mouse callback
    cv2.namedWindow('object_detection')
    cv2.setMouseCallback('object_detection', mouse_callback)

    # Continuously capture images from the camera and run inference
    while True:
        tStart = time.time()  # Start time for each loop
        im = picam2.capture_array()  
        image = cv2.resize(im, (800, 480))

        # Rotate 180 degrees if necessary (adjust if necessary)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Prepare text for display
        fps_text = f'FPS = {FPS}'  # FPS as a whole number
        confidence_text = f'Confidence = {confidence_level:.4f}'
        text_location = (right_margin, 30)  # Position at the upper right corner

        # Get system metrics
        cpu_usage, memory_usage, disk_usage, net_sent, net_recv, temperature, power = get_system_metrics()

        # Display FPS, Confidence Level, and System Metrics at the upper right corner
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, metrics_color, font_thickness, cv2.LINE_AA)
        cv2.putText(current_frame, confidence_text, (right_margin, 60), cv2.FONT_HERSHEY_DUPLEX,
                    font_size, metrics_color, font_thickness, cv2.LINE_AA)
        
        # Display system metrics in the following lines
        metrics_start_y = 90  # Start position for metrics display
        system_metrics = [
            f'CPU: {cpu_usage:.2f}%',
            f'Temp: {temperature:.2f} C',
            f'Memory: {memory_usage:.2f}%',
            f'Disk: {disk_usage:.2f}%',
            f'Net Sent: {net_sent:.2f} MB',
            f'Net Recv: {net_recv:.2f} MB',
            f'Power: {power:.2f} V'
        ]

        metrics_dict = {
            'FPS': FPS,
            'CPU (%)': cpu_usage,
            'Temperature (C)': temperature,
            'Memory (%)': memory_usage,
            'Disk (%)': disk_usage,
            'Net Sent (MB)': net_sent,
            'Net Recv (MB)': net_recv,
            'Power (V)': power,
            #'Confidence Level': confidence_level
        }

        for i, metric in enumerate(system_metrics):
            cv2.putText(current_frame, metric, (right_margin, metrics_start_y + i * 20), cv2.FONT_HERSHEY_DUPLEX,
                        font_size, metrics_color, font_thickness, cv2.LINE_AA)  # Adjusted spacing and positioning

        # Drawing the bounding boxes and labels with fixed colors
        if detection_result_list:
            for detection in detection_result_list[0].detections:
                category_name = detection.categories[0].category_name.lower()
                score = detection.categories[0].score

                # Get bounding box coordinates
                bbox = detection.bounding_box
                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)

                # Set colors based on the category (fixed for clouds and water)
                if category_name == 'cloud':
                    box_color = bright_green
                    text_color = bright_green
                elif category_name == 'water':
                    box_color = bright_blue
                    text_color = bright_blue
                else:
                    # Assign default color for unexpected categories
                    box_color = default_color
                    text_color = default_color

                # Draw bounding box
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw the label with the specified color
                cv2.putText(current_frame, f'{category_name}: {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 2)

            detection_frame = current_frame
            detection_result_list.clear()

        if detection_frame is not None:
            cv2.imshow('object_detection', detection_frame)

        # Reduce delay in cv2.waitKey to improve FPS measurement
        if cv2.waitKey(1) & 0xFF == 27:  # Adjusted to allow immediate key press detection
            break

        tEnd = time.time()
        loopTime = tEnd - tStart
        FPS = int(0.9 * FPS + 0.1 * (1 / loopTime))  # Smoothing FPS measurement

    # Save the final report once the algorithm is stopped
    save_final_report(save_folder)

    detector.close()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
#        default='cls_cloud_-_water_detection_10epoch.tflite')
        default='cls_cloud_-_water_detection_bbox_10epoch.tflite')
    parser.add_argument(
        '--maxResults',
        help='Max number of detection results.',
        required=False,
        default=5)
    parser.add_argument(
        '--scoreThreshold',
        help='The score threshold of detection results.',
        required=False,
        type=float,
        default=0.25)
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=800)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    args = parser.parse_args()

    run(args.model, int(args.maxResults),
        args.scoreThreshold, int(args.cameraId), args.frameWidth, args.frameHeight)

if __name__ == '__main__':
    main()

