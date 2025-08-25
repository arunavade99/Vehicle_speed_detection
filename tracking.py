import cv2
from ultralytics import YOLO
import ultralytics
ultralytics.checks()

import time
# import math
# from sort import *

model = YOLO('yolov8n.pt')
# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         colorsBGR = [x, y]
#         # print(colorsBGR)
#
# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rtsp://admin:xiphos@123@192.168.1.9:554/mpeg4/ch4/sub/av_stream')
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# my_file = open("coco.txt", "r")
# data = my_file.read()
# class_list = data.split("\n")

count = 0
d = 0
while True:
    ret, frame = cap.read()
    # gaussian_3 = cv2.GaussianBlur(frame, (5, 5), 2.0)
    # frame = cv2.addWeighted(frame, 2.0, gaussian_3, -0.9, 20)
    if not ret:
        break
    # count += 1
    # if count % 10 != 0:
    #     continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.track(frame, persist=True,conf=0.5, classes=[0],tracker="bytetrack.yaml",verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    if results[0].boxes.id is None:
        pass
    else:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        d=len(ids)
        for box, id in zip(boxes, ids):
            # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
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

    #     cv2.putText(frame, "Person", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.rectangle(frame, (10, 15), (260, 55), (42, 219, 151), cv2.FILLED)
    cv2.putText(frame, ('No of Persons:-') + str(d), (15, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    # if count % (fps*10)== 0:
    #     t = time.strftime("%Y-%m-%d %H-%M-%S")
    #     # t=d
    #     # if(t!=d):
    #     file = 'C:/Users/Administrator/Desktop/YOLOV8/yolov8counting-trackingvehicles-main/New folder/' + t + '.jpg'
    #     cv2.imwrite(file, frame)


    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

