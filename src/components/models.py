import math
from dataclasses import dataclass
from typing import List, Tuple, Union
from collections import deque
from src.utils import (reduce_noise_from_trajectories, exponential_moving_average, transform_points)

WINDOW = 10

@dataclass
class Track:
    def __init__(self, track_id: int, track_cls: float, track_bbox: List[int], frame_id: int) -> None:
        """
        Initializes the Track object with the provided track_id, track_cls, and track_bbox.
        
        Parameters:
            track_id (int): The ID of the track.
            track_cls (str): The class of the track.
            track_bbox (List[int]): The bounding box coordinates of the track.
        
        Returns:
            None
        """
        self.track_id = int(track_id)
        self.track_cls = int(track_cls)
        self.track_bbox = [int(val) for val in track_bbox]
        self.trail = deque(maxlen=15)
        self.speed = deque(maxlen=15)
        self.centroid = self.calculate_centroid()
        self.trail.append(self.centroid)
        self.in_zone = False
        self.last_frame = frame_id

    
    def update(self, track_bbox: List[int], track_cls: float, frame_id: int, fps: int, h_matrix: List[List]) -> None:
        self.track_bbox = [int(val) for val in track_bbox]
        self.track_cls = int(track_cls)
        cur_cx, cur_cy = self.get_centroid()
        cx, cy = self.calculate_centroid()

        # Centroid hasn't moved more than 2 pixels; for farther objects, reduces spikes in speed estimation
        # if abs(cx - cur_cx) <= 1 and abs(cy - cur_cy) <= 1:
        #     self.centroid = (cur_cx, cur_cy)
        # else:
        #     self.centroid = (cx, cy)
        self.centroid = (cx, cy)
        self.trail.append(self.centroid)
        speed = self.caculate_speed(frame_id, fps, h_matrix)
        if speed is not None:
            self.speed.append(speed)
            
        self.last_frame = frame_id
        
        
        
    def calculate_centroid(self) -> Tuple[int, int]:
        cx = int((self.track_bbox[0]+self.track_bbox[2])/2)
        # cy = int((self.track_bbox[1]+self.track_bbox[3])/2)
        # cx = int(self.track_bbox[2])
        cy = int(self.track_bbox[3])
        
        if len(self.trail) == WINDOW:
            smooth_traj = reduce_noise_from_trajectories(self.trail)
            for x,y in smooth_traj:
                self.trail.popleft()
                self.trail.append((x,y))
            
        elif len(self.trail) > WINDOW:
            prev_x, prev_y = self.trail[-1]
            cx, cy = exponential_moving_average((prev_x, prev_y), (cx, cy))       
        return (cx,cy)
    
    def caculate_speed(self, frame_id: int, fps: int, h_matrix: List[List]) -> Union[float, None]:
        if self.last_frame == frame_id or len(self.trail) < 10:
            return None
        
        pts_1 = self.trail[-2]
        pts_2 = self.trail[-1]
        
        t_pts_1 = transform_points(h_matrix, pts_1)
        t_pts_2 = transform_points(h_matrix, pts_2)
        if t_pts_1 is None or t_pts_2 is None:
            return None
        
        distance = math.sqrt(((t_pts_2[0] - t_pts_1[0])**2) + ((t_pts_2[1] - t_pts_1[1])**2))
        time_taken = abs(self.last_frame - frame_id) / fps
        
        if time_taken == 0:
            return None
        return distance / time_taken        
        
        
    
    def set_zones(self, in_zone: bool) -> None:
        self.in_zone = in_zone
        
    def get_track_id(self) -> int:
        return self.track_id
    
    def get_track_cls(self) -> int:
        return self.track_cls
    
    def get_track_bbox(self) -> List[int]:
        return self.track_bbox
    
    def get_trail(self) -> List[Tuple[int, int]]:
        return self.trail
        
    def get_centroid(self) -> Tuple[int, int]:
        return self.centroid
    
    def get_zones(self) -> bool:
        return self.in_zone
    
    def get_speed(self) -> List[float]:
        return self.speed
