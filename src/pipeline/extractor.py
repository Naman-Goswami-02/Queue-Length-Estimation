import os
import sys
import cv2
import time
import yaml
import datetime
import argparse
from scipy.signal import savgol_filter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple
from src.logger import logging
from src.exception import CustomException
from src.components.models import Track
from src.components.caching  import LRUCache
from src.utils import (load_location_data, calculate_h_matrix, draw_bouding_box,
                       transform_points, convert_seconds_to_hms)

CLASSES = ["MTW","TRW","Car","Bus","LCV","Truck","Cycle","Person"]
COLORS = [(0, 128, 255), (0, 200, 128), (0, 0, 255), (255, 255, 0), (127, 0, 255), (204, 0, 0), (0,255,255), (153, 0, 153)]
WINDOW = 5


class Detections:
    def __init__(self) -> None:
        """
        Initializes the object with an LRUCache of capacity 150.
        """
        self.cache = LRUCache(capacity=350)
    
    def update(self, detections:  List[Tuple[int, int, List[float]]], frame_id: int, fps: int, h_matrix: List[List], region: List[int]) -> None:
        """
        Updates the cache based on the detections provided.
        
        Parameters:
            detections (List[Tuple[int, int, List[float]]]): A list of detections containing track_id, class_id, and bbox.
        
        Returns:
            None
        """
        for track_id, class_id, bbox in detections:
            cx = int((bbox[0]+bbox[2])/2)
            cy = int((bbox[1]+bbox[3])/2)
            
            if cv2.pointPolygonTest(np.array(region, dtype=int), [cx, cy], False) >= 0:
                track = self.cache.get(track_id)
                if track is None:
                    self.cache.put(track_id, Track(track_id, class_id, bbox, frame_id))
                else:
                    track.update(bbox, class_id, frame_id, fps, h_matrix)
                    
                    
class VideoProcessor:
    def __init__(self, input_video: str, model_weights: str, location: str, imgsz: int, device_id: int,
                 is_save_vid: bool, skip_frames: int, speed_thres: int, x_thres: int,
                 num_thres: int, rec_freq: int):

        self.input_video = Path(input_video)
        self.model_weights = Path(r"D:\IIT Roorkee\queue-length-estimation-main\weights\best_ITD_aug.pt")
        self.location = location or self.input_video.parent.parent.stem
        self.imgsz = imgsz
        self.device_id = "cpu"
        self.is_save_vid = is_save_vid
        self.skip_frames = skip_frames
        self.speed_thres = speed_thres
        self.x_thres = x_thres
        self.num_thres = num_thres
        self.rec_freq = rec_freq
        self.file_name = self.input_video.stem
        self.detection_manager = Detections()
        self.q_len = (-1, -1, -1)
        self.last_q_time = "00:00:00"

        try:
            self.model = YOLO(self.model_weights)
            logging.info("*********************************Model loaded successfully****************************************************.")
        except Exception as e:
            logging.error(CustomException(e, sys), exc_info=True)

        self.px_pts, self.m_pts, self.detection_area = load_location_data(self.location)
        self.h_matrix = calculate_h_matrix(self.px_pts, self.m_pts)

    def calculate_q_length(self, detections: List[float]):
        total_in_zone = 0
        vehicles_in_q = []

        try:
            for track_id in detections:
                track_id = int(track_id)
                track = self.detection_manager.cache.get(track_id)
                if track is not None:
                    cx, cy = track.get_centroid()
                    class_id = track.get_track_cls()
                    speed = track.get_speed()
                    if len(speed) < WINDOW:
                        continue
                    speed = np.average(savgol_filter(speed, WINDOW, 2)) * 3.6
                    if class_id not in [6, 7] and cv2.pointPolygonTest(np.array(self.detection_area, dtype=int), [cx, cy], False) >= 0:
                        total_in_zone += 1
                        if speed <= self.speed_thres:
                            vehicles_in_q.append((track_id, cx, cy, speed, class_id))

            if len(vehicles_in_q) >= self.num_thres and len(vehicles_in_q) >= 0.5 * total_in_zone:
                vehicles_in_q = sorted(vehicles_in_q, key=lambda x: x[2])
                last_in_q = vehicles_in_q[0]
                coord = [last_in_q[1], last_in_q[2]]
                t_coord = transform_points(self.h_matrix, coord)

                if t_coord[1] >= self.q_len[0]:
                    self.q_len = (t_coord[1], coord[0], coord[1])
                    logging.info("Queue Length has been Saved Video Processded")
        except Exception as e:
            logging.error(CustomException(e, sys), exc_info=True)

    def annotate_frame(self, frame: np.ndarray, detections: List[float]) -> np.ndarray:
        annotated_frame = frame.copy()

        cv2.polylines(annotated_frame, [np.array(self.px_pts)], isClosed=True, color=(245, 135, 66), thickness=2)
        cv2.polylines(annotated_frame, [np.array(self.detection_area)], isClosed=True, color=(66, 135, 245), thickness=1)

        try:
            for track_id in detections:
                track_id = int(track_id)
                track = self.detection_manager.cache.get(track_id)
                if track is not None:
                    cx, cy = track.get_centroid()
                    class_id = track.get_track_cls()
                    x1, y1, x2, y2 = track.get_track_bbox()
                    speed = track.get_speed()

                    if cv2.pointPolygonTest(np.array(self.detection_area, dtype=int), [cx, cy], False) < 0:
                        continue

                    label = f"{track_id}"
                    if len(speed) >= WINDOW:
                        speed = np.average(savgol_filter(speed, WINDOW, 2))
                        label += f" S:{int(speed*3.6)}"
                    annotated_frame = draw_bouding_box(annotated_frame, (x1, y1), (x2, y2), (cx, cy), COLORS[class_id],
                                                       label, 0.5, 1, 1, 0.5, self.q_len)

                    trails = track.get_trail()
                    for i in range(1, len(trails)):
                        cv2.line(annotated_frame, (trails[i][0], trails[i][1]), (trails[i - 1][0], trails[i - 1][1]), COLORS[class_id], thickness=2)

            return annotated_frame
        except Exception as e:
            logging.error(CustomException(e, sys), exc_info=True)

    def process_frame(self, frame: np.ndarray, frame_id: int, fps: int) -> np.ndarray:
        try:
            results = list(self.model.track(frame, persist=True, verbose=False, device=self.device_id, imgsz=self.imgsz, tracker="botsort.yaml"))[0]

            bbox_xyxys = results.boxes.xyxy.cpu().numpy().tolist()
            class_ids = results.boxes.cls.cpu().numpy().tolist()
            tracks_ids = results.boxes.id.cpu().numpy().tolist() if results.boxes.id is not None else []

            detections = list(zip(tracks_ids, class_ids, bbox_xyxys))
            self.detection_manager.update(detections, frame_id, fps, self.h_matrix, self.detection_area)

            self.calculate_q_length(tracks_ids)
            annotated_frame = self.annotate_frame(frame, tracks_ids)

            return annotated_frame
        except Exception as e:
            logging.error(CustomException(e, sys), exc_info=True)

    def run(self):
        save_path = Path(os.getcwd()) / "data/results" / self.input_video.parent.stem
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(str(self.input_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            with tqdm(total=total_frames, desc=f"Processing video: {self.location}") as pbar:
                if self.is_save_vid:
                    out = cv2.VideoWriter(str(save_path / f"{self.file_name}.mp4"),
                                          cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps if self.skip_frames in (-1, 0) else fps // self.skip_frames,
                                          (frame_width, frame_height))

                skipped_frames = 0
                cur_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logging.warning("***************************************************************************Video has Ended and Processes Successfully*************************************************************************** .")
                        break
                    cur_frame += 1
                    if not ret:
                        break

                    if self.skip_frames != -1 and skipped_frames < self.skip_frames:
                        skipped_frames += 1
                        pbar.update(1)
                        continue
                    skipped_frames = 0

                    annotated_frame = self.process_frame(frame, cur_frame, fps)

                    seconds = cur_frame / fps
                    cur_time = convert_seconds_to_hms(seconds)
                    if datetime.datetime.strptime(cur_time, "%H:%M:%S") - datetime.datetime.strptime(self.last_q_time, "%H:%M:%S") >= datetime.timedelta(seconds=self.rec_freq):
                        self.last_q_time = cur_time
                        self.q_len = (-1, -1, -1)

                    if self.is_save_vid:
                        out.write(annotated_frame)
                    
                    if annotated_frame is None or annotated_frame.size == 0:
                        logging.warning(f"****************************************Annotated frame at {cur_frame} is None or empty. Skipping...*****************************************")
                        pbar.update(1)
                        continue
                    if annotated_frame is not None and annotated_frame.size > 0:
                        cv2.imshow("Annotated Frame", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    pbar.update(1)
        except Exception as e:
            logging.error(CustomException(e, sys), exc_info=True)
        finally:
            if self.is_save_vid:
                out.write(annotated_frame)
            if self.is_save_vid and (not out.isOpened()):
                logging.error("Failed to initialize VideoWriter. Video not saved.")

            cap.release()
            cv2.destroyAllWindows()


# --------------------------- ENTRY POINT ------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description = "QueueLength"
    )
    parser.add_argument(
        "--input_video",
        type=str,
        help="path to input video",
        default="")
    
    parser.add_argument(
        "--location",
        help="Name of the location for ROI Initialization",
        type=str,
        default="")
    
    parser.add_argument(
        "--device_id",
        help="GPU id for the Inference",
        type=str,
        default="cpu")
    
    args = parser.parse_args()
    

    config_path = Path(os.getcwd()) / "data" / "config.yaml"
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    input_video = args.input_video or config_data["input_video"]

    processor = VideoProcessor(
        input_video=input_video,
        model_weights=config_data["model_weights"],
        location=config_data["location"],
        imgsz=config_data["image_size"],
        device_id=config_data["device_id"],
        is_save_vid=config_data["is_save_vid"],
        skip_frames=config_data["skip_frames"],
        speed_thres=config_data["speed_thres"],
        x_thres=config_data["x_thres"],
        num_thres=config_data["num_thres"],
        rec_freq=config_data["record_freq"]
    )

    processor.run()
