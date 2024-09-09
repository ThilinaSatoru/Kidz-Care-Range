import os
import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from config.firebase import *
from config.paths import VIDEO_PATH, MODEL_PATH, OUTPUT_DIRECTORY, is_win

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Initialize FPS variables
low_fps = 15
high_fps = 30
fps = high_fps

# Add a boolean flag to choose input source
print('If Sample Video - 1, Camera - 2: ')
x = input()
if int(x) == 1:
    use_picamera2 = False  # Set to True to use video file, False to use PiCamera
else:
    use_picamera2 = True

if use_picamera2:
    from picamera2 import Picamera2
    import libcamera
else:
    from config.paths import VIDEO_PATH


# Motion detection logic
def motion_detected(frame1, frame2, threshold=10000):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > threshold:
            return True
    return False


def detect_objects(frame, model, conf_threshold=0.2):
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[(detections.class_id == 0) & (detections.confidence > conf_threshold)]
    return detections, results


def annotate_frame(frame, detections, results):
    labels = [
        f"{results.names[class_id]}: {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    box_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return frame


def setup_zones(frame):
    height, width, _ = frame.shape
    # Dynamically create the polygon based on the frame dimensions
    polygons = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    zones = sv.PolygonZone(polygon=polygons)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zones, color=sv.Color.RED, thickness=5, text_thickness=10,
                                             text_scale=2)
    return zones, zone_annotator


def update_firebase(detection_count):
    timestamp = int(time.time() * 1000)
    attendance = attendance_ref.get('count')[0].get('count')
    print(f"#####################################################  Attendance : {detection_count} of {attendance}.")
    if int(attendance) > int(detection_count):
        print(f"---------------------------------------------------------------- YES")
        # ###################### Push data to Firebase Realtime Database ######################
        filename = os.path.join(OUTPUT_DIRECTORY, f"img.jpg")
        cv2.imwrite(filename, frame)
        print(f"Frame saved to: {filename}")
        # Upload the image to Firebase Storage
        blob = firebase_bucket.blob(f'detections/img.jpg')
        blob.upload_from_filename(filename)
        img_url = blob.public_url  # Get the image's download URL
        # kids_ref.set({
        #     'count': detection_count,
        # })
        alert_ref.child(str(timestamp)).set({
            'type': 'range',
            'image': img_url
        })
    else:
        print(f"---------------------------------------------------------------- NO :")


def process_frame(frame, prev_frame, model, zones, zone_annotator):
    global fps
    detections, results = detect_objects(frame, model)

    mask = zones.trigger(detections=detections)
    detections_filtered = detections[mask]
    box_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
    frame = zone_annotator.annotate(scene=frame)
    detection_count = len(detections_filtered)

    if motion_detected(prev_frame, frame):
        fps = high_fps
        print(f"Motion detected, Increasing frame rate. Found:{detection_count}")
    else:
        fps = low_fps
        print(f"No motion detected, Reducing frame rate. Found:{detection_count}")

    update_firebase(detection_count)

    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # #####################################################################################
    return frame


if __name__ == "__main__":
    if use_picamera2:
        # Initialize Picamera2
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (640, 480)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.preview_configuration.controls.FrameRate = high_fps
        # Apply 180-degree rotation (both horizontal and vertical flip)
        picam2.preview_configuration.transform = libcamera.Transform(hflip=True, vflip=True)
        picam2.configure("preview")
        picam2.start()
        first_frame = picam2.capture_array()
    else:
        # Use VideoCapture for video file
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read the video")
            exit()

    zones, zone_annotator = setup_zones(first_frame)
    prev_frame = first_frame.copy()

    width, height = 640, 480  # For Picamera2
    if not use_picamera2:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_DIRECTORY + '/output.mp4', fourcc, fps, (width, height))

    try:
        if use_picamera2:
            while True:
                frame = picam2.capture_array()
                processed_frame = process_frame(frame, prev_frame, model, zones, zone_annotator)
                out.write(processed_frame)
                if is_win:
                    cv2.imshow('Processed Video', processed_frame)
                    prev_frame = frame.copy()
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                        break
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame, prev_frame, model, zones, zone_annotator)
                out.write(processed_frame)
                if is_win:
                    cv2.imshow('Processed Video', processed_frame)
                    prev_frame = frame.copy()
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                        break

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving video output...")

    finally:
        # Release resources
        if use_picamera2:
            picam2.stop()
        else:
            cap.release()
        out.release()
        if is_win == "Windows":
            cv2.destroyAllWindows()

        print(f"Video processing complete. Output saved: {OUTPUT_DIRECTORY}/output.mp4.")
