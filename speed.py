import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import sympy
import math
model = YOLO('yolov8n.pt')


def show_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coords = [x, y]
        print(mouse_coords)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', show_mouse_position)

cap = cv2.VideoCapture(r'D:\python projects\videos for project\6.mp4')

frame_count = 0
frame_times = []

line_y1 = 280
line_y2 = 330
line_y3 = 235
line_y4 = 290

line_offset = 10

avg_speed_down = 0
vehicles_down = {}
down_counter = []

avg_speed_up = 0
vehicles_up = {}
up_counter = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    detections_array = np.empty([0, 5])

    results = model.track(frame, persist=True, conf=0.2, classes=[2], tracker="bytetrack.yaml", verbose=False)
    vehicle_list = []
    detected_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    if results[0].boxes.id is None:
        pass
    else:
        vehicle_ids = results[0].boxes.id.cpu().numpy().astype(int)

    for box, vehicle_id in zip(detected_boxes, vehicle_ids):
        box_x1, box_y1, box_x2, box_y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        center_x = int(box[0] + box[2]) // 2
        center_y = int(box[1] + box[3]) // 2
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Id {vehicle_id}",
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        #####going down#####
        if 529 < center_x < 680 and line_y3 - line_offset < center_y < line_y3 + line_offset:
            cv2.line(frame, (529, line_y3), (680, line_y3), (0, 255, 0), 3)
            vehicles_down[vehicle_id] = time.time()
        if vehicle_id in vehicles_down:
            if 537 < center_x < 736 and line_y4 - line_offset < center_y < line_y4 + line_offset:
                elapsed_time = time.time() - vehicles_down[vehicle_id]
                if down_counter.count(vehicle_id) == 0:
                    down_counter.append(vehicle_id)
                    vehicles_down.pop(vehicle_id)
                    distance = 20  # meters
                    avg_speed_ms = distance / elapsed_time
                    avg_speed_down = avg_speed_ms * 3.6
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(vehicle_id), (box_x1, box_y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(avg_speed_down)) + 'Km/h', (box_x2, box_y2), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
                    cv2.line(frame, (537, line_y4), (736, line_y4), (0, 255, 0), 3)

        #####going UP#####
        if 75 < center_x < 443 and line_y2 - line_offset < center_y < line_y2 + line_offset:
            cv2.line(frame, (75, line_y2), (443, line_y2), (0, 255, 0), 3)
            vehicles_up[vehicle_id] = time.time()
        if vehicle_id in vehicles_up:
            if 162 < center_x < 456 and line_y1 - line_offset < center_y < line_y1 + line_offset:
                elapsed1_time = time.time() - vehicles_up[vehicle_id]
                if up_counter.count(vehicle_id) == 0:
                    up_counter.append(vehicle_id)
                    vehicles_up.pop(vehicle_id)
                    distance1 = 20  # meters
                    avg_speed_ms1 = distance1 / elapsed1_time
                    avg_speed_up = avg_speed_ms1 * 3.6
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(vehicle_id), (box_x1, box_y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(avg_speed_up)) + 'Km/h', (box_x2, box_y2), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
                    cv2.line(frame, (168, line_y1), (456, line_y1), (0, 255, 0), 3)

        cv2.putText(frame, str(vehicle_id), (box_x1, box_y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 2)
        # up line
        cv2.line(frame, (168, line_y1), (456, line_y1), (0, 0, 255), 1)
        cv2.line(frame, (75, line_y2), (443, line_y2), (0, 0, 255), 1)
        # down line
        cv2.line(frame, (529, line_y3), (680, line_y3), (0, 0, 255), 1)
        cv2.line(frame, (537, line_y4), (730, line_y4), (0, 0, 255), 1)

    down_count = len(down_counter)
    up_count = len(up_counter)
    cv2.putText(frame, ('goingdown:-') + str(down_count), (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    if int(avg_speed_down) > 0:
        cv2.putText(frame, ('speed:-') + str(int(avg_speed_down)) +(' km/h'), (15, 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, ('goingup:-') + str(up_count), (15, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    if int(avg_speed_up) > 0:
        cv2.putText(frame, ('speed:-') + str(int(avg_speed_up)) +(' km/h'), (15, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Display", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()