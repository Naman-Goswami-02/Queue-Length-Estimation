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
from src.utils import LRUCache, Track, load_location_data, calculate_h_matrix, draw_bouding_box, transform_points, convert_seconds_to_hms
from src.exception import CustomException
from src.logger import logging

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

def main(input_video: str, model_weights: str, location: str, imgsz: int, device_id: int, is_save_vid: bool, skip_frames: int, speed_thres: int, x_thres: int, num_thres: int, rec_freq: int) -> None:
    input_video = Path(input_video)
    logging.info(f"Video loaded from {input_video}")
    model_weights = Path(model_weights)
    file_name = input_video.stem
    detection_manager = Detections()
    last_q_time = "00:00:00"
    q_len = (-1,-1,-1)
    try:
        # model = YOLO("yolov8n.pt")
        model = YOLO(model_weights)
        logging.info(f"-----------------Model is loaded  Successfully----------------------")
    except Exception as e:
        # logging.error("Failed to load YOLO model", exc_info=True)
        logging.error(CustomException(e, sys), exc_info=True)
    
    if location is None:
        location = input_video.parent.parent.stem
    
    px_pts, m_pts, detection_area = load_location_data(location)
    h_matrix = calculate_h_matrix(px_pts, m_pts)
    
    def calculate_q_length(detections: List[float], px_pts: List[Tuple[int, int]], detection_area: List[Tuple[int, int]]) -> None:
        nonlocal detection_manager, q_len
        total_in_zone = 0 
        try:
            vehicles_in_q = []
            for track_id in detections:
                track_id = int(track_id)
                track = detection_manager.cache.get(track_id)
                if track is not None:
                    cx, cy = track.get_centroid()
                    class_id = track.get_track_cls()
                    speed = track.get_speed()
                    if len(speed) < WINDOW:
                        continue
                    # speed = np.average(speed) if len(speed) < WINDOW else np.average(savgol_filter(speed, WINDOW, 2))
                    speed = np.average(savgol_filter(speed, WINDOW, 2))
                    speed *= 3.6
                    if  class_id != 7 and class_id != 6 and cv2.pointPolygonTest(np.array(detection_area, dtype=int), [cx, cy], False) >= 0:
                        total_in_zone += 1
                        if speed <= speed_thres:
                            vehicles_in_q.append((track_id, cx, cy, speed, class_id))
                        
            
            # print(len(vehicles_in_q), total_in_zone)
            if len(vehicles_in_q) >= num_thres and len(vehicles_in_q) >= 0.50*total_in_zone:
                # vehicles_in_q = sorted(vehicles_in_q, key=lambda x: x[2], reverse=True)
                vehicles_in_q = sorted(vehicles_in_q, key=lambda x: x[2])
                logging.info(f"------------Vehicles in queue zone ---------------: {len(vehicles_in_q)} / {total_in_zone} total in zone")
                last_in_q = vehicles_in_q[0]
                coord = [last_in_q[1], last_in_q[2]]
                t_coord = transform_points(h_matrix, coord)
                
                if t_coord[1] >= q_len[0]:
                    # print(vehicles_in_q)
                    q_len = (t_coord[1], coord[0], coord[1])
            
            # print(q_len)
        
        except Exception as e:
            logging.error(CustomException(e, sys), exc_info=True)
    
    def annotate_frame(frame: np.ndarray, detections: List[float], px_pts) -> np.ndarray:
        nonlocal detection_manager
        annotated_frame = frame.copy()
        
        cv2.polylines(annotated_frame, [np.array(px_pts)], isClosed=True, color=(245,135,66), thickness=2)
        cv2.polylines(annotated_frame, [np.array(detection_area)], isClosed=True, color=(66,135,245), thickness=1)
        
        
        try:
            for track_id in detections:
                track_id = int(track_id)
                track = detection_manager.cache.get(track_id)
                if track is not None:
                    cx, cy = track.get_centroid()
                    class_id = track.get_track_cls()
                    x1,y1,x2,y2 = track.get_track_bbox()
                    speed = track.get_speed()
                    # print(speed)
                    
                    if cv2.pointPolygonTest(np.array(detection_area, dtype=int), [cx, cy], False) < 0: continue
                    
                    label = "{}".format(track_id)
                    
                    if len(speed) >= WINDOW: 
                        # label = label + f" S:{speed[0]*3.6:.2f}" if len(speed) == 1 else label + f" S:{np.average(speed)*3.6:.2f}"
                        # speed = np.average(speed) if len(speed) < WINDOW else np.average(savgol_filter(speed, WINDOW, 2))
                        speed = np.average(savgol_filter(speed, WINDOW, 2))
                        label = label + f" S: {int(speed*3.6)}"
                    annotated_frame = draw_bouding_box(annotated_frame, (x1,y1), (x2,y2), (cx, cy), COLORS[class_id], 
                                                       label, 0.5, 1, 1, 0.5, q_len)

                    trails = track.get_trail()
                    for i in range(1,len(trails)):
                        cv2.line(annotated_frame, (trails[i][0],trails[i][1]), (trails[i-1][0],trails[i-1][1]), COLORS[class_id], thickness=2)
                        
                    logging.info(f"Frame is processed and annotated")
            return annotated_frame

        except Exception as e:
            logging.error(CustomException(e, sys), exc_info=True)

    
    def process_frame(frame: np.ndarray, imgsz: int, frame_id: int, fps: int, h_matrix: np.ndarray, px_pts: List[Tuple[int, int]], detection_area: List[Tuple[int, int]]) -> np.ndarray:
        nonlocal model, detection_manager
        
        try:
            results = model.track(...)
            logging.info(f"---------------------------------Starting Processing the Frame {frame_id} ---------------------------------")
            results = list(model.track(frame, persist=True, verbose=False, device=device_id, imgsz=imgsz, tracker="botsort.yaml"))[0]
            
            bbox_xyxys = results.boxes.xyxy.cpu().numpy().tolist()
            bbox_confs = results.boxes.conf.cpu().numpy().tolist()
            class_ids = results.boxes.cls.cpu().numpy().tolist()
            
            if results.boxes.id is None:
                tracks_ids =  []
            else:
                tracks_ids = results.boxes.id.cpu().numpy().tolist()
            
            detections = list(zip(tracks_ids, class_ids, bbox_xyxys))
            
            
            detection_manager.update(detections, frame_id, fps, h_matrix, detection_area)
            
            
            # Detection in Zones
            calculate_q_length(tracks_ids, px_pts, detection_area)
            
            annotated_frame = annotate_frame(frame, tracks_ids, px_pts)
            
            return annotated_frame
        except Exception as e:
            logging.error(CustomException(e,sys), exc_info=True)
    
        
    save_path = Path(os.getcwd()) / "results" / input_video.parent.parent.parent.parent.parent.parent.stem / input_video.parent.parent.parent.parent.parent.stem / input_video.parent.parent.parent.parent.stem / input_video.parent.parent.parent.stem / input_video.parent.parent.stem  / input_video.parent.stem
    # save_path = Path(os.getcwd()) / "results" / input_video.parent.parent.parent.parent.stem  / input_video.parent.parent.parent / input_video.parent.parent / input_video.parent.stem

    os.makedirs(save_path, exist_ok=True)
    print(str(input_video))
    cap = cv2.VideoCapture(str(input_video))
    logging.info(f"---------------------------------Video is Successfully opened---------------------------------: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        with tqdm(total=total_frames, desc=f"Processing video: {location}") as pbar:
            if is_save_vid:
                if skip_frames == -1 or skip_frames == 0:
                    out = cv2.VideoWriter(save_path / f"{file_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
                    skip_frames = -1
                else:
                    out = cv2.VideoWriter(save_path / f"{file_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps//skip_frames, (frame_width, frame_height))
            
            skipped_frames = 0
            cur_frame = 0 
            while cap.isOpened():
                ret, frame = cap.read()
                cur_frame += 1
                logging.info(f"Processing frame {cur_frame}/{total_frames}")
                if not ret:
                    break


                if skip_frames != -1 and skipped_frames < skip_frames:
                    skipped_frames += 1
                    pbar.update(1)
                    logging.info(f"Skipping the frame {cur_frame} due to skip_frames={skip_frames}")
                    continue

                if skip_frames != -1:
                    skipped_frames = 0

                # frame = cv2.resize(frame, (1080,1080))
                annotated_frame = process_frame(frame, imgsz, cur_frame, fps, h_matrix, px_pts, detection_area)

                seconds = cur_frame / fps
                cur_time = convert_seconds_to_hms(seconds)
                if datetime.datetime.strptime(cur_time, "%H:%M:%S") - datetime.datetime.strptime(last_q_time, "%H:%M:%S") >= datetime.timedelta(seconds=rec_freq):
                    last_q_time = cur_time
                    q_len = (-1,-1,-1)
                
                if is_save_vid:
                    out.write(annotated_frame)
                
                cv2.imshow("Annotated Frame", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                pbar.update(1)
        
    except Exception as e:
        logging.error(CustomException(e,sys), exc_info=True)
        print(CustomException(e, sys)) 
    finally:  
        if is_save_vid:
            out.release()
        cap.release()
        cv2.destroyAllWindows()


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video",
        type=str,
        help="path to input video",
        default=""
        )
    
    args = parser.parse_args()
    
    config_path = Path(os.getcwd()) / "data" / "config.yaml"
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    input_video = config_data["input_video"]
    model_weights = config_data["model_weights"]
    location = config_data["location"]
    imgsz = config_data["image_size"]
    device_id = config_data["device_id"]
    is_save_vid = config_data["is_save_vid"]
    skip_frames = config_data["skip_frames"]
    speed_thres = config_data["speed_thres"]
    x_thres = config_data["x_thres"]
    num_thres = config_data["num_thres"]
    rec_freq = config_data["record_freq"]
    
    input_video = args.input_video
    
    main(
        input_video,
        model_weights,
        location,
        imgsz,
        device_id,
        is_save_vid,    
        skip_frames,
        speed_thres,
        x_thres,
        num_thres,
        rec_freq
    )