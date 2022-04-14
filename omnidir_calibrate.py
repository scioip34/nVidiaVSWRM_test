"""
Script to calibrate ultra-wide FOV fisheye lenses using opencv-contrib-python's omnilib submodule.

Running this script needs opencv-contrib-python instead of the normal opencv-python to get the omnidir module.
Before installing it via 'pip install opencv-contrib-python',
uninstall other cv2 versions via 'pip uninstall opencv-python'.

This script only works for cylindrical reconstruction. For other rectification types the new_K camera matrix needs to be
defined differently.

Calibration can happen via jpg images in the data folder or if the calibration points have been previously extracted
and saved as npz files, via npz files from the data folder.

created by mezdahun
14.04.2022
"""
import json
import signal
import cv2
import glob
import numpy as np
from contextlib import contextmanager

assert cv2.__version__[0] == '4', 'The fisheye module requires opencv version >= 3.0.0'


# Checkerboard stuff is prone to get lazy, we introduce timeout
class TimeoutException(Exception): pass


# Signal handler for timeout
def signal_handler(signum, frame):
    raise TimeoutException("Timed out!")


# Connecting alarm signals to handler
signal.signal(signal.SIGALRM, signal_handler)


# Introducing possible context for using timeout
@contextmanager
def time_limit(seconds):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


# Checkerboard dimensions (inner crossing points)
CHECKERBOARD = (6, 9)
# Calibration stopping criteria
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000, 0.001)
# Object points for calibration
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# If previously found calibration points are already exported as npz files
pointsets = glob.glob('./data/calib_pointset_*.npz')
# If npz files are not found but jpg calibration images are available
images = glob.glob('./data/*.jpg')

if len(pointsets) == 0:
    # Did not found npz files
    for fname in images:
        print(f"Processing image: ", fname)
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        print("Finding Checkerboard...")
        try:
            with time_limit(3):
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        except TimeoutException as e:
            ret = False
            print("Timed out!")

        # If found, add object points, image points (after refining them)
        if ret == True:
            # print("Found calibration points!")
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
        else:
            print("Didn't find calibration points!")
else:
    print("Found previously exported pointsets, will load calibration points from these!")
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _img_shape = img.shape[:2]
    for pointset_path in pointsets:
        print(f"Loading points from {pointset_path}")
        calibration_points = np.load(pointset_path)
        imgPointsArray = list(calibration_points["imgPointsArray"].astype(np.float32))
        imgpoints.extend(imgPointsArray)
        objectPointsArray = list(calibration_points["objectPointsArray"].astype(np.float32))
        objpoints.extend(objectPointsArray)
    newobjpoints = []
    for objp in objpoints:
        newobjpoints.append(np.expand_dims(objp, axis=0))
    objpoints = newobjpoints

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rms, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    size=gray.shape[::-1],
    K=None, xi=None, D=None, rvecs=rvecs, tvecs=tvecs,
    flags=cv2.omnidir.CALIB_FIX_K1,
    criteria=subpix_criteria)

print("Found " + str(N_OK) + " valid data points for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

DIM = _img_shape[::-1]
img = cv2.imread(images[0])
dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
assert dim1[0] / dim1[1] == DIM[0] / DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in " \
                                             "calibration "

# dim2 is the desired image size after undistortion
dim2 = (2000, 1000)

# Edge size is to include all the image and reduce cv2 rectification on the edges. This should be adjusted according
# to dim2. If too much is visible on the edges (double images) you should reduce it, if too less (rectified image)
# increase it.
edge_size = 200
# New desired camera matrix that changes according to rectification type (e.g. cylindrical)
new_K = np.array([[(dim2[0] - 2 * edge_size) / 3.1415, 0, (dim2[0] - edge_size)],
                  [0.0, (dim2[1]) / 3.1415, dim2[1] / 2],
                  [0.0, 0.0, 1.0]])

# Rotation matrix to be applied within remapping
R = np.array([[1, 0, 0],
              [0, 0, -1],
              [0, 1, 0]], dtype=np.float32)

# Data summary
data = {'dim1': dim1,
        'dim2': dim2,
        'K': np.asarray(K).tolist(),
        'D': np.asarray(D).tolist(),
        'xi': np.asarray(xi).tolist(),
        'new_K': np.asarray(new_K).tolist(),
        'R': np.asarray(R).tolist()}

with open("fisheye_calibration_data.json", "w") as f:
    json.dump(data, f)

# Creating and saving maps
map1, map2 = cv2.omnidir.initUndistortRectifyMap(K, D, xi, R, new_K, dim2, cv2.CV_32FC1,
                                                 cv2.omnidir.RECTIFY_CYLINDRICAL)
np.savez("maps.npz", map1=map1, map2=map2)

# Maps can be created on any hardware and be loaded later on the edge where opencv-contrib is not available
# path_to_maps = "maps.npz"
# maps = np.load(path_to_maps)
# map1 = maps['map1']
# map2 = maps['map2']

# Remapping original image
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Alternative to init and remap but much slower and needs to have opencv-contrib installed @param flags Flags
# indicates the rectification type,  RECTIFY_PERSPECTIVE, RECTIFY_CYLINDRICAL, RECTIFY_LONGLATI and
# RECTIFY_STEREOGRAPHIC undistorted_img = cv2.omnidir.undistortImage(img, K, D, xi, cv2.omnidir.RECTIFY_CYLINDRICAL,
# Knew=new_K, R=R, new_size = dim2)

# Showing original image
cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.imshow("original", img)

# Showing undistorted image
cv2.namedWindow("undistorted", cv2.WINDOW_NORMAL)
cv2.imshow("undistorted", undistorted_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
