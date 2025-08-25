import cv2
from ultralytics import YOLO
import ultralytics
ultralytics.checks()

import time
import datetime

def run_person_tracking(video_source=0, model_path='yolov8n.pt'): # 0 for webcam, or provide video file path
    """
    Runs real-time person tracking using YOLO and displays results in a window.
    Args:
        video_source: Camera index or video file path.
        model_path: Path to YOLO model weights.
    """
    ultralytics.checks()
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            with open("camera_missing.txt", "a") as f:
                f.write(f"[{datetime.datetime.now()}] Camera feed missing.\n")
            break

        frame = cv2.resize(frame, (1020, 500))
        tracking_results = model.track(
            frame,
            persist=True,
            conf=0.5,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False
        )
        detected_boxes = tracking_results[0].boxes.xyxy.cpu().numpy().astype(int) # 0 for person class
        person_count = 0
        if tracking_results[0].boxes.id is not None:
            person_ids = tracking_results[0].boxes.id.cpu().numpy().astype(int)
            person_count = len(person_ids)
            for bounding_box, person_id in zip(detected_boxes, person_ids):
                center_x = int(bounding_box[0] + bounding_box[2]) // 2
                center_y = int(bounding_box[1] + bounding_box[3]) // 2
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"Id {person_id}",
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        # Overlay person count
        cv2.rectangle(frame, (10, 15), (260, 55), (42, 219, 151), cv2.FILLED)
        cv2.putText(
            frame,
            f'No of Persons: {person_count}',
            (15, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imshow("Person Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_person_tracking()

