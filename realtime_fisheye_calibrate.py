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
    framerate=1,
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

# Define the chess board rows and columns
rows = 5
cols = 3

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objectPointsArray = []
imgPointsArray = []
rets = []
if os.path.isfile("./calib_points_realtime.npz"):
    print("Found calibration points from previous runs! Loading them!")
    calibration_points = np.load("./calib_points_realtime.npz")
    imgPointsArray = list(calibration_points["imgPointsArray"])
    objectPointsArray = list(calibration_points["objectPointsArray"])
    rets = list(calibration_points["rets"])

data_pattern = "./data/*.jpg"



def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=2))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        frame_id = 0
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            img_to_show = img.copy()

            if len(imgPointsArray) > 0:
                for corn_i in range(len(imgPointsArray)):
                    cv2.drawChessboardCorners(img_to_show, (rows, cols), imgPointsArray[corn_i], rets[corn_i])

            # Display the image
            cv2.imshow('CSI Camera', img_to_show)

            # This also acts as
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                # ESC pressed
                print("'q' was hit, closing...")
                break
            elif k == ord("s"):
                # s pressed
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_name = f"VSWRM_img_{frame_id}.jpg"
                cv2.imwrite(img_name, img)
                frame_id += 1
                print(f"{img_name} saved!")
            elif k == ord("c"):
                print("finding grid and saving image points")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (rows, cols), cv2.CALIB_CB_FAST_CHECK)

                if ret:
                    # Refine the corner position
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    print("Found grid corners...")
                    # Add the object points and the image points to the arrays
                    objectPointsArray.append(objectPoints)
                    imgPointsArray.append(corners)
                    rets.append(ret)
                    print("Calibraiton pointset size: ", len(imgPointsArray))

                    # Draw the corners on the image
                    cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
                    # Display the image
                    cv2.imshow('CSI Camera', img)

            elif k == ord("p"):
                print("finished collecting data, closing CV first...")
                break

        cap.release()
        cv2.destroyAllWindows()

        # Saving Calibration points
        print("saving calibration points...")
        np.savez('calib_points_realtime.npz', imgPointsArray=imgPointsArray, objectPointsArray=objectPointsArray, rets=rets)

        # Calibrate the camera and save the results
        print("Calculating calibration matrices...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1],
                                                           None, None)
        np.savez('calib_realtime.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

        # Print the camera calibration error
        print("Saved calibration matrices, calculating calibration error...")
        error = 0
        for i in range(len(objectPointsArray)):
            imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
            error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

        print("Total error: ", error / len(objectPointsArray))

        print("Bye")
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()

# # Loop over the image files
# for path in glob.glob(data_pattern):
#     # Load the image and convert it to gray scale
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
#
#     # Make sure the chess board pattern was found in the image
#     if ret:
#         # Refine the corner position
#         corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#
#         # Add the object points and the image points to the arrays
#         objectPointsArray.append(objectPoints)
#         imgPointsArray.append(corners)
#
#         # Draw the corners on the image
#         cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
#
#     # Display the image
#     cv2.imshow('chess board', img)
#     cv2.waitKey(500)
#
# # Calibrate the camera and save the results
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
# np.savez('./data/calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
#
# # Print the camera calibration error
# error = 0
#
# for i in range(len(objectPointsArray)):
#     imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
#     error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)
#
# print("Total error: ", error / len(objectPointsArray))
#
# # Load one of the test images
# img = cv2.imread(glob.glob(data_pattern)[0])
# h, w = img.shape[:2]
#
# # Obtain the new camera matrix and undistort the image
# newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
#
# # Crop the undistorted image
# # x, y, w, h = roi
# # undistortedImg = undistortedImg[y:y + h, x:x + w]
#
# # Display the final result
# cv2.imshow('chess board', undistortedImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()