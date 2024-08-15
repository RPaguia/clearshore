import cv2
import numpy as np
from picamera2 import Picamera2
import time
import psutil
import subprocess
import pandas as pd
import os

# Initialize Picamera2
picam2 = Picamera2()

# Set display width and height for the camera
dispW = 1280
dispH = 720

# Set camera resolution
picam2.preview_configuration.main.size = (1920, 1080)  # Set resolution to 1920x1080

# Configure camera settings
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 30
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Initialize variables for FPS calculation
fps = 0
prev_time = time.time()

# Desired display size for the touchscreen
touchscreen_width = 800
touchscreen_height = 480

# Create a window named "Camera"
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

# Global variables for trackbar values
hueLow, hueHigh, satLow, satHigh, valLow, valHigh = 97, 115, 0, 65, 0, 255

def onTrack1(val):
    global hueLow
    hueLow = val
    print('Hue Low', hueLow)
    
def onTrack2(val):
    global hueHigh
    hueHigh = val
    print('Hue High', hueHigh)

def onTrack3(val):
    global satLow
    satLow = val
    print('Sat Low', satLow)

def onTrack4(val):
    global satHigh
    satHigh = val
    print('Sat High', satHigh)

def onTrack5(val):
    global valLow
    valLow = val
    print('Val Low', valLow)

def onTrack6(val):
    global valHigh
    valHigh = val
    print('Val High', valHigh)

# Create trackbars for adjusting HSV values
cv2.createTrackbar("Hue Low", "Camera", hueLow, 179, onTrack1)
cv2.createTrackbar("Hue High", "Camera", hueHigh, 179, onTrack2)
cv2.createTrackbar("Sat Low", "Camera", satLow, 255, onTrack3)
cv2.createTrackbar("Sat High", "Camera", satHigh, 255, onTrack4)
cv2.createTrackbar("Val Low", "Camera", valLow, 255, onTrack5)
cv2.createTrackbar("Val High", "Camera", valHigh, 255, onTrack6)

# Function to get system metrics
def get_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=None)
    temp = subprocess.check_output("vcgencmd measure_temp", shell=True)
    temp = float(temp.decode("UTF-8").strip().replace("temp=", "").replace("'C", ""))
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    net_io = psutil.net_io_counters()
    net_sent_mb = net_io.bytes_sent / (1024 * 1024)
    net_recv_mb = net_io.bytes_recv / (1024 * 1024)
    voltage = subprocess.check_output("vcgencmd measure_volts", shell=True)
    voltage = float(voltage.decode("UTF-8").strip().replace("volt=", "").replace("V", ""))
    return cpu_usage, temp, memory_usage, disk_usage, net_sent_mb, net_recv_mb, voltage

# Initialize DataFrame to store the results
columns = [
    "Filename", "Subfolder", "HueLow", "HueHigh", "SatLow", "SatHigh", "ValLow", "ValHigh",
    "CPU_Usage", "Temperature", "Memory_Usage", "Disk_Usage", "Net_Sent_MB", "Net_Recv_MB", "Power",
    "FPS"
]
results_list = []

# Directory to save the reports and snapshots
save_directory = "/home/skySentinel/CloudTracker.HSV"
snapshot_directory = os.path.join(save_directory, "snapshots_hsv")
if not os.path.exists(snapshot_directory):
    os.makedirs(snapshot_directory)

# Function to generate a unique filename based on the current time and thresholds
def generate_filename(prefix, extension):
    thresholds = f"({hueLow},{hueHigh})({satLow},{satHigh})({valLow},{valHigh})"
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{thresholds}_{timestamp}.{extension}"

# Capture and save a snapshot of each frame
def save_snapshots(display_image, myMaskSmall, myObjectSmall):
    camera_filename = os.path.join(snapshot_directory, generate_filename("Camera", "png"))
    mymask_filename = os.path.join(snapshot_directory, generate_filename("myMask", "png"))
    myobject_filename = os.path.join(snapshot_directory, generate_filename("myObject", "png"))

    cv2.imwrite(camera_filename, display_image)
    cv2.imwrite(mymask_filename, myMaskSmall)
    cv2.imwrite(myobject_filename, myObjectSmall)
    
    print(f"Snapshots saved as {camera_filename}, {mymask_filename}, {myobject_filename}")

# Finalize and save the report
def save_report():
    results_df = pd.DataFrame(results_list, columns=columns)
    csv_filename = os.path.join(save_directory, generate_filename("cloud_detection_metrics_with_system", "csv"))
    excel_filename = os.path.join(save_directory, generate_filename("cloud_detection_metrics_with_system", "xlsx"))
    results_df.to_csv(csv_filename, index=False)
    results_df.to_excel(excel_filename, index=False)
    print(f"Report saved as {csv_filename} and {excel_filename}")

# Mouse callback function to handle clicks
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        display_image, myMaskSmall, myObjectSmall = param
        save_snapshots(display_image, myMaskSmall, myObjectSmall)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        save_report()

# Set the mouse callback function
cv2.setMouseCallback("Camera", mouse_callback, param=None)

# Main loop to capture and display video
total_start_time = time.time()
while True:
    # Capture image
    im = picam2.capture_array()
    
    # Calculate the new width of the displayed window based on the current width of the "Camera" frame
    window_width = cv2.getWindowImageRect("Camera")[2]
    
    # Resize image to fit the new width of the "Camera" window while maintaining aspect ratio
    height, width, _ = im.shape
    new_height = int(height * (window_width / width))
    im_resized = cv2.resize(im, (window_width, new_height))
    
    # Position the resized video frame within the "Camera" window
    display_image = im_resized
    
    # Get system metrics
    cpu_usage, temp, memory_usage, disk_usage, net_sent_mb, net_recv_mb, voltage = get_system_metrics()
    
    # Calculate FPS after system metrics are collected to avoid delay issues
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    
    # Add FPS and system metrics text to the image in the "Camera" frame (reduced size)
    font_scale = 0.4  # Reduce font size to 40% of the previous size
    cv2.putText(display_image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 100), 1)
    cv2.putText(display_image, f"CPU: {cpu_usage:.2f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 100), 1)
    cv2.putText(display_image, f"Temp: {temp:.2f}C", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 100), 1)
    cv2.putText(display_image, f"Mem: {memory_usage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 100), 1)
    cv2.putText(display_image, f"Disk: {disk_usage:.2f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 100), 1)
    cv2.putText(display_image, f"Net Sent: {net_sent_mb:.2f}MB", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 100), 1)
    cv2.putText(display_image, f"Net Recv: {net_recv_mb:.2f}MB", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 100), 1)
    cv2.putText(display_image, f"Power: {voltage:.2f}V", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 100), 1)
    
    # Convert to HSV for masking and object detection
    lowerBound = np.array([hueLow, satLow, valLow])
    upperBound = np.array([hueHigh, satHigh, valHigh])
    frameHSV = cv2.cvtColor(im_resized, cv2.COLOR_BGR2HSV)
    myMask = cv2.inRange(frameHSV, lowerBound, upperBound)
    myMaskSmall = cv2.resize(myMask, (int(dispW / 4), int(dispH / 4)))
    myObject = cv2.bitwise_and(im_resized, im_resized, mask=myMask)
    myObjectSmall = cv2.resize(myObject, (int(dispW / 4), int(dispH / 4)))
    
    # Append results to the list
    results_list.append({
        "Filename": "Current Capture",  # Change this if filenames are being processed
        "Subfolder": "Live Capture",    # Adjust this if using specific subfolders
        "HueLow": hueLow, "HueHigh": hueHigh, "SatLow": satLow, "SatHigh": satHigh, "ValLow": valLow, "ValHigh": valHigh,
        "CPU_Usage": cpu_usage, "Temperature": temp, "Memory_Usage": memory_usage, "Disk_Usage": disk_usage,
        "Net_Sent_MB": net_sent_mb, "Net_Recv_MB": net_recv_mb, "Power": voltage, "FPS": fps
    })
    
    # Display the image in a window named "Camera"
    cv2.imshow("Camera", display_image)
    cv2.imshow('my Mask', myMaskSmall)
    cv2.imshow('myObject', myObjectSmall)
    
    # Update the mouse callback parameter with the current display image
    cv2.setMouseCallback("Camera", mouse_callback, param=(display_image, myMaskSmall, myObjectSmall))
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

total_end_time = time.time()
print(f"Total Processing Time: {total_end_time - total_start_time:.4f} seconds")

# Release resources
cv2.destroyAllWindows()
