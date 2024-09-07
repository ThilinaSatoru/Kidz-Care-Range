import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from config.paths import VIDEO_PATH, MODEL_PATH, OUTPUT_DIRECTORY
from config.firebase import *

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Initialize FPS variables
low_fps = 15
high_fps = 30
fps = high_fps


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


def process_frame(frame, prev_frame, model, zones, zone_annotator):
    global fps
    detections, results = detect_objects(frame, model)

    mask = zones.trigger(detections=detections)
    detections_filtered = detections[mask]

    detection_count = len(detections_filtered)
    # ###################### Push data to Firebase Realtime Database ######################
    # users_ref.push({
    #     'date': date_now.date().isoformat(),
    #     'time': date_now.time().isoformat(),
    #     'detection': detection_count,
    #     'prediction': ''
    # })
    # #####################################################################################
    box_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
    frame = zone_annotator.annotate(scene=frame)

    if motion_detected(prev_frame, frame):
        fps = high_fps
        print(f"Motion detected, Increasing frame rate. Found:{detection_count}")
    else:
        fps = low_fps
        print(f"No motion detected, Reducing frame rate. Found:{detection_count}")

    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        exit()
    zones, zone_annotator = setup_zones(first_frame)
    prev_frame = first_frame.copy()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_DIRECTORY + '/output.mp4', fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame = process_frame(frame, prev_frame, model, zones, zone_annotator)
            out.write(processed_frame)
            # Display the frame
            cv2.imshow('Processed Video', processed_frame)
            prev_frame = frame.copy()

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving the video output...")

    finally:
        # Release everything properly
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video processing complete. Output saved : {OUTPUT_DIRECTORY}/output.mp4.")
