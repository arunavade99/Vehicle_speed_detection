import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import sympy
import math
model = YOLO('yolov8n.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# cap = cv2.VideoCapture('veh2.mp4')
cap = cv2.VideoCapture('6.mp4')


# my_file = open("coco.txt", "r")
# data = my_file.read()
# class_list = data.split("\n")
# print(class_list)

count = 0
tcount=[]

cy1 = 280
cy2 = 330
cy3=235
cy4=290

offset = 10

a_speed_kh=0
vh_down = {}
counter = []

a_speed_kh1=0
vh_up = {}
counter1 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    detections = np.empty([0, 5])

    results = model.track(frame, persist=True, conf=0.2, classes=[2], tracker="bytetrack.yaml", verbose=False)
    list = []
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    if results[0].boxes.id is None:
        pass
    else:
        ids = results[0].boxes.id.cpu().numpy().astype(int)

    for box, id in zip(boxes, ids):
        x3, y3, x4, y4 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cx = int(box[0] + box[2]) // 2
        cy = int(box[1] + box[3]) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Id {id}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )


        #####going down#####
        # if cy1 < (cy + offset) and cy1 > (cy - offset):
        if 529 < cx < 680 and cy3 - offset < cy < cy3 + offset:
            cv2.line(frame, (529, cy3), (680, cy3), (0, 255, 0), 3)

            vh_down[id] = time.time()
        if id in vh_down:

            # if cy2 < (cy + offset) and cy2 > (cy - offset):
            if 537 < cx < 736 and cy4 - offset < cy < cy4 + offset:
                elapsed_time = time.time() - vh_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    vh_down.pop(id)
                    distance = 20  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    # print(a_speed_kh, 'km/h')
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
                    cv2.line(frame, (537, cy4), (736, cy4), (0, 255, 0), 3)


        #####going UP#####
        if 75 < cx < 443 and cy2 - offset < cy < cy2 + offset:
        # if cy2 < (cy + offset) and cy2 > (cy - offset):
            cv2.line(frame, (75, cy2), (443, cy2), (0, 255, 0), 3)

            vh_up[id] = time.time()
        if id in vh_up:

            if 162 < cx < 456 and cy1 - offset < cy < cy1 + offset:
            # if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed1_time = time.time() - vh_up[id]

                if counter1.count(id) == 0:
                    counter1.append(id)
                    vh_up.pop(id)
                    distance1 = 20  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
                    cv2.line(frame, (168, cy1), (456, cy1), (0, 255, 0), 3)


        # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 2)
        # up line
        cv2.line(frame, (168, cy1), (456, cy1), (0, 0, 255), 1)
        cv2.line(frame, (75, cy2), (443, cy2), (0, 0, 255), 1)
        # down line
        cv2.line(frame, (529, cy3), (680, cy3), (0, 0, 255), 1)
        cv2.line(frame, (537, cy4), (730, cy4), (0, 0, 255), 1)
        # cv2.line(frame, (179, cy1), (417, cy1), (0, 0, 255), 1)
        # cv2.line(frame, (100, cy2), (408, cy2), (0, 0, 255), 1)


    # # cv2.putText(frame, ('L1'), (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    #
    # cv2.line(frame, (100, cy2), (408, cy2), (0, 255, 255), 1)
    #
    # # cv2.putText(frame, ('L2'), (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    d = (len(counter))
    u = (len(counter1))
    # e=(len(tcount))
    # print("jfuyfluy",e)
    cv2.putText(frame, ('goingdown:-') + str(d), (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    if int(a_speed_kh) >0:
        cv2.putText(frame, ('speed:-') + str(int(a_speed_kh)) +(' km/h'), (15, 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    # print(str(int(a_speed_kh)))
    cv2.putText(frame, ('goingup:-') + str(u), (15, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    if int(a_speed_kh1) >0:
        cv2.putText(frame, ('speed:-') + str(int(a_speed_kh1)) +(' km/h'), (15, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

