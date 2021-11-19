import datetime
import logging
import os
from math import floor

import cv2
import numpy as np
import time

from tflite_runtime.interpreter import Interpreter


import cv2

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=25,
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

def output_tensor(interpreter, i):
    """Returns dequantized output tensor if quantized before."""
    output_details = interpreter.get_output_details()[i]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)


MODEL_NAME = './edgetpu'
GRAPH_NAME = 'model_test_fullinteger.tflite'
LABELMAP_NAME = 'labelmap.txt'
USE_TPU = False
INTQUANT = True
# it takes a little longer on the first run and then runs at normal speed.
import random
import glob

if USE_TPU:
    pkg = importlib.util.find_spec('tflite_runtime')
    if not pkg:
        from tensorflow.lite.python.interpreter import load_delegate
    else:
        from tflite_runtime.interpreter import load_delegate

min_conf_threshold = 0.25

resW, resH = 1280, 720
imW, imH = int(resW), int(resH)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if USE_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print(width, height)

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

print('Model loaded!')

# if monitoring.SAVE_CNN_TRAINING_DATA:
#     training_data_folder = os.path.join(monitoring.SAVED_VIDEO_FOLDER, 'training_data')
#     ROBOT_NAME = os.getenv('ROBOT_NAME', 'Robot')
#     EXP_ID = os.getenv('EXP_ID', 'expXXXXXX')
#     if os.path.isdir(training_data_folder):
#         shutil.rmtree(training_data_folder)
#     os.makedirs(training_data_folder, exist_ok=True)


# Wait a certain number of seconds to allow the camera time to warmup
print('Waiting 8 secs for camera warmup!')
time.sleep(8)


cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    # Window
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        ret_val, img = cap.read()
        cv2.imshow("CSI Camera", img)
        # This also acts as
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unable to open camera")


# frame_id = 0
# for frame in picam.capture_continuous(raw_capture,
#                                         format=camera.CAPTURE_FORMAT,
#                                         use_video_port=camera.USE_VIDEO_PORT):
#     # Grab the raw NumPy array representing the image
#     if camera.FLIP_CAMERA:
#         img = cv2.flip(frame.array, -1)
#     else:
#         img = frame.array

#     # Clear the raw capture stream in preparation for the next frame
#     raw_capture.truncate(0)

#     # Adding time of capture for delay measurement
#     capture_timestamp = datetime.utcnow()

#     # Collecting training data in a predefined freq
#     if frame_id == 0:
#         CNN_TD_last_collect = capture_timestamp

#     # clear vision stream if polluted to avoid delay
#     t0 = capture_timestamp

#     frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     frame_resized = cv2.resize(frame_rgb, (width, height))
#     input_data = np.expand_dims(frame_resized, 0).astype('float32')

#     # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
#     if floating_model:
#         input_data = (np.float32(input_data) - input_mean) / input_std
#     if INTQUANT:
#         input_data = input_data.astype('uint8')

#     t1 = datetime.utcnow()
#     logger.info(f'preprocess time {(t1 - t0).total_seconds()}')
#     # Perform the actual detection by running the model with the image as input
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()

#     # Bounding box coordinates of detected objects
#     boxes = interpreter.get_tensor(output_details[0]['index'])[0]
#     # Class index of detected objects
#     # classes = interpreter.get_tensor(output_details[1]['index'])[0]
#     # Confidence of detected objects
#     scores = interpreter.get_tensor(output_details[2]['index'])[0]

#     # Dequantize if input and output is int quantized
#     if INTQUANT:
#         scale, zero_point = output_details[0]['quantization']
#         boxes = scale * (boxes - zero_point)

#         # scale, zero_point = output_details[1]['quantization']
#         # classes = scale * (classes - zero_point)

#         scale, zero_point = output_details[2]['quantization']
#         scores = scale * (scores - zero_point)

#     t2 = datetime.utcnow()
#     delta = (t2 - t1).total_seconds()
#     logger.debug(f"Inference time: {delta}, rate={1 / delta}")  #

#     blurred = np.zeros([img.shape[0], img.shape[1]])

#     for i in range(len(boxes)):
#         if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
#             # if scores[i] == np.max(scores):
#             # Get bounding box coordinates and draw box
#             # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
#             ymin = int(max(1, (boxes[i, 0] * imH)))
#             xmin = int(max(1, (boxes[i, 1] * imW)))
#             ymax = int(min(imH, (boxes[i, 2] * imH)))
#             xmax = int(min(imW, (boxes[i, 3] * imW)))

#             blurred[ymin:ymax, xmin:xmax] = 255
#             cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
#             img = cv2.putText(img, f'score={scores[i]:.2f}', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.5, (255, 0, 0), 2, cv2.LINE_AA)

#     t3 = datetime.utcnow()
#     logger.debug(f"Postprocess time: {(t3 - t1).total_seconds()}")

#     # Forwarding result to VPF extraction
#     logger.debug(f'Queue length{raw_vision_stream.qsize()}')
#     high_level_vision_stream.put((img, blurred, frame_id, capture_timestamp))
#     t4 = datetime.utcnow()
#     logger.debug(f'Transferring time: {(t4 - t3).total_seconds()}')

#     # Collecting training data for CNN fine tune if requested
#     if monitoring.SAVE_CNN_TRAINING_DATA:
#         if (capture_timestamp - CNN_TD_last_collect).total_seconds() > 1/monitoring.CNN_TRAINING_DATA_FREQ:
#             frame_name = f'{EXP_ID}_{ROBOT_NAME}_CNNTD_frame{frame_id}.png'
#             frame_path = os.path.join(training_data_folder, frame_name)
#             cv2.imwrite(frame_path, frame_rgb)
#             CNN_TD_last_collect = capture_timestamp

#     # Forwarding result for visualization if requested
#     if visualization_stream is not None:
#         visualization_stream.put((img, blurred, frame_id))

#     # To test infinite loops
#     if env.EXIT_CONDITION:
#         break

#     t5 = datetime.utcnow()
#     logger.info(f'total vision_rate: {1 / (t5 - t0).total_seconds()}')
