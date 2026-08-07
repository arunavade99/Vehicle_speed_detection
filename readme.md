# Vehicle Speed Detection

This project uses YOLOv8 and OpenCV to track objects and estimate vehicle speed in video input.

## Files

- `speed.py`: detects vehicles, tracks them, and estimates speed for vehicles moving in both directions.
- `tracking.py`: tracks objects (people by default) using a unique ID and displays the count in real time.
- `yolov8n.pt`: YOLOv8 model weights used by both scripts.
- `requirement.txt`: Python package dependencies.

## Setup (Conda)

1. Create a new conda environment:
   ```bash
   conda create -n vehicle_speed python=3.11 -y
   ```
2. Activate the environment:
   ```bash
   conda activate vehicle_speed
   ```
3. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

## Run

- To run vehicle speed detection:
  ```bash
  python speed.py
  ```

- To run object/person tracking:
  ```bash
  python tracking.py
  ```

## Notes

- `speed.py` expects a video source path hardcoded in the script. Update the `cap = cv2.VideoCapture(...)` line if you want to use a different video or webcam.
- `tracking.py` uses `yolov8n.pt` by default and tracks class `0` (person) with ByteTrack.
- Press `q` in the display window to stop the script.
