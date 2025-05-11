import cv2
import numpy as np

class CourtMapper:
    def __init__(self, src_points, dst_points):
        self.M = cv2.getPerspectiveTransform(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32)
        )

    def warp_point(self, point):
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        p2 = cv2.perspectiveTransform(p, self.M)
        return tuple(p2[0][0].tolist())