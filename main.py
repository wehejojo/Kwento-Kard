import cv2
import numpy as np
from cv2 import aruco

def apply_translation(H, offset):
    tx, ty = offset
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    return T @ H

marker = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
marker_params = aruco.DetectorParameters()
aug_img = cv2.imread("A-imgs/UCoLab.png")

def imageMapper(frame, src, marker_corners, scale=1.5, offset_ratio=(1.1, 0)):
    src_h, src_w = src.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    dst = marker_corners[0].astype(np.float32)
    top_left, top_right, bottom_right, bottom_left = dst

    right_vec = top_right - top_left
    marker_width = np.linalg.norm(right_vec)

    aspect_ratio = src_h / src_w
    scaled_width = marker_width * scale
    scaled_height = scaled_width * aspect_ratio

    unit_right = right_vec / np.linalg.norm(right_vec)
    down_vec = bottom_left - top_left
    unit_down = down_vec / np.linalg.norm(down_vec)
    unit_perpendicular = np.array([-unit_right[1], unit_right[0]])  # 90 degrees CCW

    offset_vec = unit_right * (marker_width * offset_ratio[0]) + unit_down * (marker_width * offset_ratio[1])

    new_top_left = top_right + offset_vec
    new_top_right = new_top_left + unit_right * scaled_width
    new_bottom_right = new_top_right + unit_perpendicular * scaled_height
    new_bottom_left = new_top_left + unit_perpendicular * scaled_height

    dst_points = np.array([new_top_left, new_top_right, new_bottom_right, new_bottom_left], dtype=np.float32)
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)

    H, _ = cv2.findHomography(src_points, dst_points)
    warped_image = cv2.warpPerspective(src, H, (frame_w, frame_h))

    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_points.astype(np.int32), 255)

    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=inv_mask)
    foreground = cv2.bitwise_and(warped_image, warped_image, mask=mask)

    cv2.polylines(foreground, [dst_points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

    result = cv2.add(background, foreground)
    return result


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    detector = aruco.ArucoDetector(marker, marker_params)
    corners, ids, _ = detector.detectMarkers(img)

    if corners:
        print(f"Aruco ID detected: {ids.flatten()}")
        for corner_set in corners:
            img = imageMapper(img, aug_img, corner_set, scale=3.5)
            # for corner in corner_set:
            #     corner = corner.reshape(-1, 1, 2)
            #     img = imageMapper(img, aug_img, corner)

    cv2.imshow('moment', img)

    if cv2.waitKey(1) == ord("0"):
        break

cap.release()
cv2.destroyAllWindows()
