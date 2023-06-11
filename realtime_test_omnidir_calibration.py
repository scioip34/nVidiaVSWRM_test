'''
 Based on the following tutorial:
   https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/cameraCalibration/cameraCalibration.py
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
'''
import numpy as np
import cv2
import glob
import os


# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
        capture_width=1640,
        capture_height=1232,
        display_width=1640,
        display_height=1232,
        framerate=20,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


# # Define the chess board rows and columns
# rows = 6
# cols = 9
#
# # Set the termination criteria for the corner sub-pixel algorithm
# criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
#
# # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
# objectPoints = np.zeros((rows * cols, 3), np.float32)
# objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
#
# # Create the arrays to store the object points and the image points
# objectPointsArray = []
# imgPointsArray = []
# rets = []
#
# calibration_data = np.load("./calib_realtime.npz")
# mtx = calibration_data["mtx"]
# dist = calibration_data["dist"]
# rvecs = calibration_data["rvecs"]
# tvecs = calibration_data["tvecs"]
#
# # Obtain the new camera matrix and undistort the image
# w = 1640
# h = 1232
# DIM = (w, h)
# newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMtx, (w, h), 5)

# dim2 = (2000, 1000)
#
# K = np.array([[1293.910853909818, -0.7089112955623251, 851.4440016613135],
#               [0.0, 1293.215575443015, 623.1688566855132],
#               [0.0, 0.0, 1.0]])
#
# D = np.array([[0.0,
#                0.6298377541803084,
#                -0.0026918796324268733,
#                0.0011223958104573642]])
#
# new_K = np.array([[(dim2[0]-400)/3.1415, -0.86, (dim2[0]-200)],
#           [0.0, (dim2[1])/3.1415, dim2[1]/2],
#           [0.0, 0.0, 1.0]])
#
# R = np.array([[1, 0, 0],
#               [0, 0, -1],
#               [0, 1, 0]], dtype=np.float32)
#
# xi = np.array([[2.1990889469537933]])

# print("Calculating maps...")
# map1, map2 = cv2.omnidir.initUndistortRectifyMap(K, D, xi, R, new_K, dim2, cv2.CV_32FC1, cv2.omnidir.RECTIFY_CYLINDRICAL)

maps = np.load("maps.npz")
map1 = maps['map1']
map2 = maps['map2']

cv2.namedWindow("CSI Camera", cv2.WINDOW_NORMAL)

def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=2))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        frame_id = 0
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()

            if frame_id > 200:
                # undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
                undistortedImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                # Display the image
                cv2.imshow('CSI Camera', undistortedImg)

            # This also acts as
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                # ESC pressed
                print("'q' was hit, closing...")
                break
            elif k == ord("s"):
                # s pressed
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_name = f"distVSWRM_img_{frame_id}.jpg"
                cv2.imwrite(img_name, img)
                img_name = f"undistVSWRM_img_{frame_id}.jpg"
                cv2.imwrite(img_name, undistortedImg)
                frame_id += 1
                print(f"{img_name} saved!")
            elif k == ord("e"):
                print("Empty video buffer...")
                not_empty = True
                while not_empty:
                    not_empty = cap.grab()
                print("Buffer emptied!")

            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
        print("Bye")
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()