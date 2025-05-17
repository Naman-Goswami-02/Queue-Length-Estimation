import os
import cv2
import sys
import yaml
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from src.exception import CustomException
from scipy.signal import savgol_filter
from typing import List, Tuple, Union, Optional
from src.logger import logging
WINDOW = 10


def calculate_h_matrix(px_roi, m_roi) -> np.ndarray:
    # matrix = cv2.getPerspectiveTransform(np.array(px_roi, dtype=np.float32), np.array(m_roi, dtype=np.float32))
    matrix, _ = cv2.findHomography(np.array(px_roi, dtype=np.float32), np.array(m_roi, dtype=np.float32), cv2.RANSAC, 5.0)
    return matrix


def convert_seconds_to_hms(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def draw_bouding_box(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], centroid: Tuple[int, int], color: Tuple[int,int,int], label: str, font_scale: float, thickness_bbox: int, thickness_label: int, scaling_factor_font: float, q_len: Tuple[int], roundness: float = 0.6) -> np.ndarray:
    
    annotated_image = image.copy()
    try:

        radius = (
            int((bottom_right[0]-top_left[0]) // 2*roundness)
            if abs(top_left[0] - bottom_right[0]) < abs(top_left[1] - bottom_right[1])
            else int((bottom_right[1]-top_left[1]) // 2*roundness)
        )    

        
        p1 = top_left
        p2 = (bottom_right[0], top_left[1])
        p3 = bottom_right
        p4 = (top_left[0], bottom_right[1])
        
        
        cv2.line(annotated_image, (p1[0] + radius,p1[1]), (p2[0] - radius,p2[1]), color=color, thickness=thickness_bbox, lineType=cv2.LINE_AA)
        cv2.line(annotated_image, (p2[0],p2[1] + radius), (p3[0],p3[1] - radius), color=color, thickness=thickness_bbox, lineType=cv2.LINE_AA)
        cv2.line(annotated_image, (p4[0] + radius, p4[1]), (p3[0] - radius, p3[1]), color=color, thickness=thickness_bbox, lineType=cv2.LINE_AA)
        cv2.line(annotated_image, (p1[0],p1[1] + radius), (p4[0], p4[1] - radius), color=color, thickness=thickness_bbox, lineType=cv2.LINE_AA)
        
        cv2.ellipse(annotated_image, (p1[0] + radius, p1[1] + radius), (radius, radius), 180, 0, 90, color=color, thickness=thickness_bbox, lineType=cv2.LINE_AA)
        cv2.ellipse(annotated_image, (p2[0] - radius, p2[1] + radius), (radius, radius), 270, 0, 90, color=color, thickness=thickness_bbox, lineType=cv2.LINE_AA)
        cv2.ellipse(annotated_image, (p3[0] - radius, p3[1] - radius), (radius, radius), 0, 0, 90, color=color, thickness=thickness_bbox, lineType=cv2.LINE_AA)
        cv2.ellipse(annotated_image, (p4[0] + radius, p4[1] - radius), (radius, radius), 90, 0, 90, color=color, thickness=thickness_bbox, lineType=cv2.LINE_AA)
        
        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=thickness_label)[0]
        c2 = p1[0] + radius + t_size[0] + int(2 * scaling_factor_font), p1[1] + t_size[1] + int(5 * scaling_factor_font)
        cv2.rectangle(annotated_image, (p1[0] + radius, p1[1]), c2, color, -1, cv2.LINE_AA)
        cv2.putText(annotated_image, label, (p1[0] + radius, c2[1] - int(5 * scaling_factor_font)), 0, font_scale, (0,0,0), thickness=thickness_label, lineType=cv2.LINE_AA)
        cv2.circle(annotated_image, centroid, int(5 * scaling_factor_font), color, cv2.FILLED)
        
        anchor = (10,10)
        q_label = f"Q_len: {q_len[0]}"
        t_size = cv2.getTextSize(q_label, 0, fontScale=font_scale, thickness=thickness_label)[0]
        c2 = anchor[0] + t_size[0] + int(2 * scaling_factor_font), anchor[1] + t_size[1] + int(5 * scaling_factor_font)
        cv2.rectangle(annotated_image, (anchor[0], anchor[1]), c2, (125,213,118), -1, cv2.LINE_AA)
        cv2.putText(annotated_image, q_label, (anchor[0], c2[1] - int(5 * scaling_factor_font)), 0, font_scale, (0,0,0), thickness=thickness_label, lineType=cv2.LINE_AA)
        
        if q_len[0] != -1:
            cv2.circle(annotated_image, (q_len[1],q_len[2]), int(10 * scaling_factor_font), (125,213,118), cv2.FILLED)
            # cv2.line(annotated_image, (0,q_label[2]), (1080,q_label[2]), (125,213,118), thickness=2, lineType=cv2.LINE_AA)
        
        return annotated_image
    except Exception as e:
        raise CustomException(e, sys)


def transform_points(h_matrix: List[List], point_in_pixel: Tuple[int, int]) -> Union[None, Tuple[float, float]]:
    px_coords = np.float32([point_in_pixel[0], point_in_pixel[1], 1])
    m_coords = np.dot(h_matrix, px_coords)
    m_coords_normalized = m_coords/m_coords[2]

    return (m_coords_normalized[0], m_coords_normalized[1])


def load_location_data(location: str) -> Optional[Tuple[List[int], List[float], List[int]]]:
    try:
        data_path = Path(os.getcwd()) / "data" / "locations/locations.yaml"
        
        with open(data_path, "r") as file:
            data = yaml.safe_load(file)
            
        if data is not None and data.get(location) is None:
            raise CustomException(f"No data found for the location {location}", sys)

        px_pts, m_pts, detection_area = data[location]["roi_px"], data[location]["roi_m"], data[location]["detection_area"]
        
        return px_pts, m_pts, detection_area
    except Exception as e:
        logging.error(CustomException(f"Location Data Loading Failed [{location}]", sys), exc_info=True)
            

def exponential_moving_average(prev_point: float, new_point: float, aplha=0.6) -> float:
    x = aplha * new_point[0] + (1 - aplha) * prev_point[0]
    y = aplha * new_point[1] + (1 - aplha) * prev_point[1]
    return int(x), int(y)

def reduce_noise_from_trajectories(trajectories: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    x_traj, y_traj = zip(*trajectories)
    
    x_traj = np.array(savgol_filter(np.array(x_traj), WINDOW, 2), dtype=int)
    y_traj = np.array(savgol_filter(np.array(y_traj), WINDOW, 2), dtype=int)
    
    trajectories = list(zip(x_traj, y_traj))
    
    return trajectories                              
            




