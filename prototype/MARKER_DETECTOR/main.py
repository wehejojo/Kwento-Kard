import cv2
import numpy as np
from cv2 import aruco

marker = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_params = aruco.DetectorParameters()

def imageMapper(img):
    return

cap = cv2.VideoCapture(0) 
while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(marker, marker_params)

    corners, ids, reject = detector.detectMarkers(img)
    print(ids)

    if corners:
        for id, corner in zip(ids, corners):
            cv2.polylines(img, [corner.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)

            corners = corner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners  

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

    cv2.imshow('moment', img)
    
    if (key_pressed := cv2.waitKey(1)) == ord("0"):
        break

cap.release()
cv2.destroyAllWindows()