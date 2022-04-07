"""
Rinning this script needs opencv-contrib-python instead of the normal opencv-python to get the omnidir module.
Before installing via pip uninstall other cv2 versions via pip uninstall opencv-python.
"""
import cv2

assert cv2.__version__[0] == '4', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import glob
import signal

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

def signal_handler(signum, frame):
    raise TimeoutException("Timed out!")
signal.signal(signal.SIGALRM, signal_handler)

@contextmanager
def time_limit(seconds):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
pointsets = glob.glob('./data/calib_pointset_*.npz')
images = glob.glob('./data/*.jpg')
if len(pointsets) == 0:
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
            with time_limit(1):
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        except TimeoutException as e:
            ret = False
            print("Timed out!")

        # If found, add object points, image points (after refining them)
        if ret == True:
            # print("Found calibration points!")
            print(objp.dtype)
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
        else:
            print("Didn't find calibration points!")
else:
    print("Found previously exported pointsets, will load calibration points from these")
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _img_shape = img.shape[:2]
    for pointset_path in pointsets:
        print(f"Loading points from {pointset_path}")
        calibration_points = np.load(pointset_path)
        print(calibration_points["objectPointsArray"].dtype)
        imgPointsArray = list(calibration_points["imgPointsArray"].astype(np.float32))
        imgpoints.extend(imgPointsArray)
        objectPointsArray = list(calibration_points["objectPointsArray"].astype(np.float32))
        objpoints.extend(objectPointsArray)
    newobjpoints = []
    for objp in objpoints:
        newobjpoints.append(np.expand_dims(objp, axis=0))
    objpoints = newobjpoints

N_OK = len(objpoints)
print(len(objpoints))
print(objpoints[0].shape)
print(imgpoints[0].shape)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
# rms, _, _, _, _ = \
#     cv2.fisheye.calibrate(
#         objpoints,
#         imgpoints,
#         gray.shape[::-1],
#         K,
#         D,
#         rvecs,
#         tvecs,
#         calibration_flags,
#         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
#     )
# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
# xi = np.zeros(1)
# rms, _, _, _, _, _ = \
#     cv2.omnidir.calibrate(
#         objpoints,
#         imgpoints,
#         gray.shape[::-1],
#         K,
#         xi,
#         D,
#         calibration_flags,
#         (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
#         rvecs,
#         tvecs
#     )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rms, K, xi, D, rvecs, tvecs, idx  =  cv2.omnidir.calibrate(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            size=gray.shape[::-1],
            K=None, xi=None, D=None,
            flags=cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER,
            criteria=subpix_criteria)

print("Found " + str(N_OK) + " valid data points for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

DIM = _img_shape[::-1]

# DIM=(1640, 1232)
# K=np.array([[630.1409550612487, 0.0, 820.0], [0.0, 630.1631329394177, 616.0], [0.0, 0.0, 1.0]])
# D=np.array([[-0.006845095123421685, 0.0007987746751401334, 1.7327659142422208e-05, 0.00016330173727187834]])

balance = 1
dim2 = None
dim3 = None

img = cv2.imread(images[3])
dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
assert dim1[0] / dim1[1] == DIM[0] / DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in " \
                                             "calibration "
if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document
# failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_CONSTANT)

data = {'dim1': dim1,
        'dim2': dim2,
        'dim3': dim3,
        'K': np.asarray(K).tolist(),
        'D': np.asarray(D).tolist(),
        'new_K': np.asarray(new_K).tolist(),
        'scaled_K': np.asarray(scaled_K).tolist(),
        'balance': balance}

import json

with open("fisheye_calibration_data.json", "w") as f:
    json.dump(data, f)

cv2.imshow("original", img)

cv2.imshow("undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
