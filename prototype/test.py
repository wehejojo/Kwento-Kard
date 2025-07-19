import cv2
import numpy as np
import urllib
from selenium import webdriver
from cv2 import aruco
import time

marker = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
marker_params = aruco.DetectorParameters()

# Function to capture screenshot of a website
def capture_website_screenshot(url):
    options = webdriver.ChromeOptions()
    options.add_argument('headless')  # To run the browser in headless mode (without GUI)
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    # Adjust the window size if needed
    driver.set_window_size(1024, 768)
    # Wait for the website to load (adjust this time as needed)
    time.sleep(2)
    # Capture screenshot
    screenshot = driver.get_screenshot_as_png()
    driver.quit()
    # Convert the screenshot to OpenCV image format
    img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), -1)
    return img

def imageOffset(H, translation):
    translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
    return np.dot(translation_matrix, H)

def imageMapper(frame, src, points):
    src_height, src_width = src.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    points_list = [(int(pt[0][0]), int(pt[0][1])) for pt in points]

    src_points = np.array([[0, 0], [src_width/3, 0], [src_width/3, src_height/2], [0, src_height/2]])
    H, _ = cv2.findHomography(srcPoints=src_points, dstPoints=np.array([points_list]))

    offset = (150, -25)
    H = imageOffset(H, offset)

    warped_image = cv2.warpPerspective(src, H, (frame_width, frame_height))

    border_points = np.array([[0, 0], [src_width - 1, 0], [src_width - 1, src_height - 1], [0, src_height - 1]])
    border_points = np.array(border_points, dtype=np.int32)

    cv2.polylines(src, [border_points], isClosed=True, color=(0, 255, 0), thickness=4)

    if frame.shape == warped_image.shape: 
        result = cv2.addWeighted(frame, 1, warped_image, 1, 0)
    else:
        result = frame

    return result

# Example URL of the website to capture


cap = cv2.VideoCapture(0) 
while cap.isOpened():
    _, img = cap.read()

    if not _:
        break

    detector = aruco.ArucoDetector(marker, marker_params)

    corners, ids, reject = detector.detectMarkers(img)

    website_url = "https://wehejojo.github.io/CC14-WebAppDev"
    website_img = capture_website_screenshot(website_url)

    if corners:
        for corner_set in corners:
            for corner in corner_set:
                corner = corner.reshape(-1, 1, 2)
                img = imageMapper(img, website_img, corner)

    cv2.imshow('moment', img)
    
    if cv2.waitKey(1) == ord("0"):
        break

cap.release()
cv2.destroyAllWindows()
