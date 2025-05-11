import cv2
import numpy as np
import constants
from utils import (
    convert_pixel_to_meters,
    measure_distance,
    get_center_of_bbox  # you may define this in utils
)

class MiniCourt:
    """
    Draws a bird's-eye mini-court overlay in the frame corner and maps
    player/ball positions into mini-court coordinates.
    """
    def __init__(self, frame_shape, buffer=50, width=250, height=500, padding=20):
        h, w = frame_shape[:2]
        # define overlay rectangle in top-left corner
        self.end_x = w - buffer
        self.end_y = buffer + height
        self.start_x = self.end_x - width
        self.start_y = self.end_y - height
        self.padding = padding
        # define drawable court region
        self.court_start = (self.start_x + padding, self.start_y + padding)
        self.court_end   = (self.end_x   - padding, self.end_y   - padding)
        self.court_width = self.court_end[0] - self.court_start[0]
        # precompute court line endpoints in pixels based on constants
        self.keypoints = self._compute_drawing_keypoints()
        self.lines = self._define_lines()

    def _compute_drawing_keypoints(self):
        # returns 14 (x,y) points flattened: order TL, TR, top sideline midpoints, etc.
        pts = [None]*28
        # top-left corner
        pts[0:2] = list(self.court_start)
        # top-right corner
        pts[2:4] = [self.court_end[0], self.court_start[1]]
        # bottom-left
        pts[6:8] = [self.court_start[0], self.court_end[1]]
        # bottom-right
        pts[4:6] = [self.court_end[0],   self.court_end[1]]
        # more points based on real dimensions in constants
        def to_px(m): return int(m * (self.court_width / constants.COURT_LENGTH_M))
        # center service line points
        mid_x = self.court_start[0] + to_px(constants.COURT_LENGTH_M/2)
        pts[12:14] = [mid_x, self.court_start[1]]
        pts[14:16] = [mid_x, self.court_end[1]]
        # net midpoints
        pts[24:26] = [self.court_start[0], self.court_start[1] + (self.court_end[1]-self.court_start[1])//2]
        pts[26:28] = [self.court_end[0],   self.court_start[1] + (self.court_end[1]-self.court_start[1])//2]
        # fill remaining based on constants.SINGLes
        # ... (add inside if needed)
        return pts

    def _define_lines(self):
        # pairs of point indices to connect
        return [(0,2),(4,6),(0,4),(2,6),(12,14),(24,26)]

    def draw(self, frame):
        # draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.start_x,self.start_y), (self.end_x,self.end_y), (255,255,255), cv2.FILLED)
        alpha = 0.5
        frame[self.start_y:self.end_y, self.start_x:self.end_x] = (
            cv2.addWeighted(overlay, alpha, frame, 1-alpha,0)
            [self.start_y:self.end_y, self.start_x:self.end_x]
        )
        # draw court lines
        for i,j in self.lines:
            pt1 = (int(self.keypoints[i*2]), int(self.keypoints[i*2+1]))
            pt2 = (int(self.keypoints[j*2]), int(self.keypoints[j*2+1]))
            cv2.line(frame, pt1, pt2, (0,0,0), 2)
        # draw keypoints
        for k in range(0,len(self.keypoints),2):
            cv2.circle(frame, (int(self.keypoints[k]),int(self.keypoints[k+1])), 4, (0,0,255), -1)
        return frame

    def map_positions(self, player_boxes, ball_centers, court_kpts):
        """
        Convert each frame's detections into mini-court coordinates.
        player_boxes: list of dicts [{'bbox':(...),'conf':...},...]
        ball_centers: list of (x,y) or None
        court_kpts: list of 14 (x,y) real-frame keypoints
        returns two lists: players_mini, ball_mini
        """
        players_mini = []
        ball_mini    = []
        # compute homography per frame
        for dets, ball, real_kpts in zip(player_boxes, ball_centers, court_kpts):
            src = np.array(real_kpts[:4],dtype=np.float32)
            dst = np.array([self.keypoints[i*2:i*2+2] for i in range(4)],dtype=np.float32)
            H,_ = cv2.findHomography(src, dst)
            ppos = {}
            for idx, d in enumerate(dets):
                x1,y1,x2,y2 = d['bbox']
                cen = np.array([[(x1+x2)/2,(y1+y2)/2]],dtype=np.float32).reshape(-1,1,2)
                m = cv2.perspectiveTransform(cen,H)[0][0]
                ppos[idx] = tuple(m.tolist())
            players_mini.append(ppos)
            if ball:
                cen = np.array([[ball]],dtype=np.float32).reshape(-1,1,2)
                m = cv2.perspectiveTransform(cen,H)[0][0]
                ball_mini.append(tuple(m.tolist()))
            else:
                ball_mini.append(None)
        return players_mini, ball_mini