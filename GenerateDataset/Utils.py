import numpy as np
import math
import random


class Utils:
    @staticmethod
    def get_random_crop_box(W, H, sample_img_size):
        left = random.randint(0, W-sample_img_size[0])
        top = random.randint(0, H-sample_img_size[1])
        return [left, top, left+sample_img_size[0], top+sample_img_size[1]]

    @staticmethod
    def check_intersects(a, b):
        x1, y1, xx1, yy1 = a
        x2, y2, xx2, yy2 = b
        return not (xx1 < x2 or x1 > xx2 or yy1 < y2 or y1 > yy2)

    @staticmethod
    def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        """
        angle *= np.pi/180
        ox, oy = origin
        px, py = point
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return int(qx), int(qy)
