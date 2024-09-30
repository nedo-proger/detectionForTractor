# pylint: disable=wrong-import-position
import os
import sys
import cv2
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
import torch


BASE_DIRECTORY = Path(__file__).parent
YOLO_DIRECTORY = os.path.abspath(os.path.join(BASE_DIRECTORY))
sys.path.append(YOLO_DIRECTORY)

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
OBJECTS = ["cattle", "fallen_tree", "human", "power_line", "tractor"]

from models.experimental import attempt_load
from models.yolo import Model
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


class YoloDetector:
    def __init__(self, model_file: str, use_converted_model=False,
                 model_config=Path(BASE_DIRECTORY, "cfg/deploy/yolov7.yaml"), image_scaling=640):
        path_to_model = Path(BASE_DIRECTORY, model_file)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if use_converted_model:
            raw_model = torch.load(path_to_model, map_location=self.device)
            self.model = Model(cfg=model_config)
            self.model.eval()
            self.model.load_state_dict(raw_model)
            self.model.to(self.device)
        else:
            self.model = attempt_load(path_to_model, map_location=self.device)

        self.image_scaling = image_scaling

    def detect(self, image: np.ndarray, mapper: Callable = lambda x: x) -> Tuple[List[np.ndarray], List[Any], Any]:
        img0 = letterbox(image, self.image_scaling, 32)[0]  # Масштабированное изображение
        img = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(img)[0]

        predictions = non_max_suppression(predictions)
        img_labels_list = []
        detected_objects_coords = []
        class_labels = []

        for _, det in enumerate(predictions):
            if len(det):
                # Используем размеры img0 для масштабирования координат
                coords = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for coord, class_label in zip(coords, det[:, 5]):
                    y0, x0, y1, x1 = coord.int().tolist()  # Преобразуем в целые значения
                    img_labels_list.append(image[int(y0): int(y1), int(x0): int(x1)])  # Изменено
                    detected_objects_coords.append(coord)
                    class_labels.append(mapper(class_label.item()))
        return img_labels_list, class_labels, detected_objects_coords


def CameraDetect():
    detector1 = YoloDetector("best.pt")
    cap_cam = cv2.VideoCapture(0)

    if cap_cam.isOpened():
        print("WIDTH = ", cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("HEIGHT = ", cap_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS = ", cap_cam.get(cv2.CAP_PROP_FPS))

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap_cam.read()
        if ret:
            new_frame_time = cv2.getTickCount()

            img_labels_list, class_labels, detected_objects_coords = detector1.detect(frame)

            for i in range(len(img_labels_list)):
                buff = detected_objects_coords[i].tolist()
                coords = [int(b) for b in buff]
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), COLORS[int(class_labels[i])], 3)
                cv2.putText(frame, OBJECTS[int(class_labels[i])], (coords[0], coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[int(class_labels[i])], 2, cv2.LINE_AA)

            # Рассчитать FPS
            fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Отобразить FPS на кадре с плавающей точкой (например, 29.97)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)

            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap_cam.release()
    cv2.destroyAllWindows()


def VideoDetect(path: str):
    cap = cv2.VideoCapture(path)

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("WIDTH = ", width)
        print("HEIGHT = ", height)
        print("FPS = ", fps)

    detector1 = YoloDetector("best.pt")
    video_writer = cv2.VideoWriter(path + "_detected.avi", cv2.VideoWriter_fourcc(*'XVID'), fps / 5, (width, height))
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            arr = detector1.detect(frame)

            for i in range(len(arr[0])):
                buff = arr[2][i].tolist()
                coords = [int(b) for b in buff]
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), COLORS[int(arr[1][i])], 3)
                cv2.putText(frame, OBJECTS[int(arr[1][i])], (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[int(arr[1][i])], 2, cv2.LINE_AA)

            video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


CameraDetect()
