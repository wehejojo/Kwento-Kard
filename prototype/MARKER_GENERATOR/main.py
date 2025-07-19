import cv2, random
import numpy as np
from cv2 import aruco

marker = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
MARKER_SIZE = 400

for i in range(1, 51):
    MARKER_ID = random.randint(0, 49)
    aruco_image = aruco.generateImageMarker(marker, MARKER_ID, MARKER_SIZE)
    cv2.imshow('moment', aruco_image)
    cv2.imwrite(f"MARKER_GENERATOR/markers/aruco-{i}.jpg", aruco_image)
# cv2.waitKey(0)