#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import time
import ctypes
import argparse
import numpy as np
import tensorrt as trt
import datetime

import pycuda.driver as cuda
import pycuda.autoinit

from image_batcher import ImageBatcher
from visualize import visualize_detections

import cv2

def gstreamer_pipeline(
    capture_width=320,
    capture_height=200,
    display_width=320,
    display_height=200,
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

class TensorRTInfer:
    """
    Implements inference for the Model TensorRT engine.
    """

    def __init__(self, engine_path, preprocessor):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        self.preprocessor = preprocessor
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        print(f"Batch size: {self.batch_size}")
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch, scales=None, nms_threshold=None):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """

        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])

        # Process the results
        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        classes = outputs[3]
        detections = []
        normalized = (np.max(boxes) < 2.0)
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                # Depending on preprocessor, box scaling will be slightly different.
                if self.preprocessor == "fixed_shape_resizer":
                    scale_x = self.inputs[0]['shape'][1] if normalized else 1.0
                    scale_y = self.inputs[0]['shape'][2] if normalized else 1.0

                    if scales and i < len(scales):
                        scale_x /= scales[i][0]
                        scale_y /= scales[i][1]
                    if nms_threshold and scores[i][n] < nms_threshold:
                        continue
                    detections[i].append({
                        'ymin': boxes[i][n][0] * scale_y,
                        'xmin': boxes[i][n][1] * scale_x,
                        'ymax': boxes[i][n][2] * scale_y,
                        'xmax': boxes[i][n][3] * scale_x,
                        'score': scores[i][n],
                        'class': int(classes[i][n]),
                    })
                elif self.preprocessor == "keep_aspect_ratio_resizer":
                    scale = self.inputs[0]['shape'][2] if normalized else 1.0
                    if scales and i < len(scales):
                        scale /= scales[i]
                    if nms_threshold and scores[i][n] < nms_threshold:
                        continue
                    detections[i].append({
                        'ymin': boxes[i][n][0] * scale,
                        'xmin': boxes[i][n][1] * scale,
                        'ymax': boxes[i][n][2] * scale,
                        'xmax': boxes[i][n][3] * scale,
                        'score': scores[i][n],
                        'class': int(classes[i][n]),
                    })
        return detections

if __name__ == "__main__":

    min_score = 0.25
    engine_path = "../floatinputmodel.trt"
    preprocessor = "fixed_shape_resizer"
    labels = ["VSWRM_bot"]

    trt_infer = TensorRTInfer(engine_path, preprocessor)

    print(gstreamer_pipeline(flip_method=2))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():

        # only works with batch_size 1 from here
        frame_id = 0
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            # capture
            ret_val, img = cap.read()

            # resize
            img = cv2.resize(img, (320, 320))

            # connect with old implementation
            batch = np.array([img])
            images = [frame_id]
            scales = [[1.0, 1.0]]

            # inference
            t0 = datetime.datetime.now()
            detections = trt_infer.infer(batch, scales)
            t1 = datetime.datetime.now()
            print(f"inf time: {(t1 - t0).total_seconds()}")

            # visualize detections


            cv2.imshow("CSI Camera", img)
            # This also acts as
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                # ESC pressed
                print("'q' was hit, closing...")
                break

        print()
        print("Finished Processing")
