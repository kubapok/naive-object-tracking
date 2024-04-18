import sys
import cv2

import numpy as np
from pydantic import BaseModel
from typing import Tuple

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


MODEL_NAME = "yolov8m.pt"
MAX_FRAMES_WO_RECOGNITION = 3
CONF_THRESHOLD = 0.4
MAX_OBJECT_MOVEMENT_RATIO_X = 0.1
MAX_OBJECT_MOVEMENT_RATIO_Y = 0.1


input_video_path = sys.argv[1]
output_video_path = 'object_tracker_output.avi'

model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(input_video_path)
assert cap.isOpened(), "Error reading video file"


width, height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
output_video = cv2.VideoWriter(output_video_path,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (width, height))
class TrackerFrame(BaseModel):
    frame_id: int
    box_center: Tuple[int, int]

class Tracker(BaseModel):
    class_name: str
    frames: list[TrackerFrame,]

MAX_ACCEPTED_X_SHIFT = width * MAX_OBJECT_MOVEMENT_RATIO_X
MAX_ACCEPTED_Y_SHIFT = width * MAX_OBJECT_MOVEMENT_RATIO_Y

def get_or_create_tracker(trackers: dict, class_name: str, box_center: Tuple[int, int]) -> Tracker:
    tracker: Tracker = None

    for tracker_id in trackers.keys():
        if (class_name == trackers[tracker_id].class_name and
                (box_center[0] - trackers[tracker_id].frames[-1].box_center[0]) < MAX_ACCEPTED_X_SHIFT and
                (box_center[1] - trackers[tracker_id].frames[-1].box_center[1]) < MAX_ACCEPTED_Y_SHIFT and
                frame_counter - trackers[tracker_id].frames[-1].frame_id < MAX_FRAMES_WO_RECOGNITION):
            tracker = trackers[tracker_id]
            break

    if tracker is None:
        id = 0 if not trackers.keys() else max(trackers.keys()) + 1
        trackers[id] = Tracker(class_name=class_name, frames=list())
        tracker = trackers[max(trackers.keys())]

    return tracker

trackers = dict()

frame_counter = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame)[0]
    boxes = results.boxes.xyxy.cpu()

    if len(results.boxes.cls) > 0:

        for i in range(len(boxes)):
            cls = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])
            box = results.boxes[i].xyxy.cpu()[0]

            if conf < CONF_THRESHOLD:
                continue

            annotator = Annotator(frame, line_width=2)
            class_name = model.model.names[int(cls)]
            annotator.box_label(box, color=colors(int(cls), True), label=class_name)

            box_center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

            tracker = get_or_create_tracker(trackers, class_name, box_center)

            cv2.circle(frame, (box_center), 7, colors(int(cls), True), -1)
            tracker.frames.append(TrackerFrame(frame_id=frame_counter, box_center=box_center))
            points = np.array([t.box_center for t in tracker.frames], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

    output_video.write(frame)
    frame_counter += 1

output_video.release()
cap.release()

print(f'The output video is saved to {output_video_path}')
