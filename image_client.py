#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import datetime
from functools import partial
import os
import sys
import cv2

from PIL import Image
import numpy as np
from attrdict import AttrDict

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from pprint import pprint

from visualize import visualize_detections_live_triton

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()

def gstreamer_pipeline(
        capture_width=320,
        capture_height=200,
        display_width=320,
        display_height=200,
        framerate=30,
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

# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    #print("BAKED RESPONSE: ", result.get_response().id)
    user_data._completed_requests.put((result, error))


FLAGS = None


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    # Summarizing model
    pprint(f"Model inputs: {model_metadata.inputs}")
    pprint(f"Model outputs: {model_metadata.outputs}")

    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    #if len(model_metadata.outputs) != 1:
    #    raise Exception("expecting 1 output, got {}".format(
    #        len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        #elif dim > 1:
        #    non_one_cnt += 1
        #    if non_one_cnt > 1:
        #        raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
        (input_config.format != mc.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " +
                        mc.ModelInput.Format.Name(input_config.format) +
                        ", expecting " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                        " or " +
                        mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, c, h, w, input_config.format,
            input_metadata.datatype)


def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """

    output_array = results.as_numpy(output_name)
    #if len(output_array) != batch_size:
    #    raise Exception("expected {} results, got {}".format(
    #        batch_size, len(output_array)))

    # Include special handling for non-batching models
    for results in output_array:
        print(results)
        #if not batching:
        #    results = [results]
        #for result in results:
        #    if output_array.dtype.type == np.object_:
        #        cls = "".join(chr(x) for x in result).split(':')
        #    else:
        #        cls = result.split(':')
        #    print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))


def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):
    protocol = FLAGS.protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    # outputs = [
    #     client.InferRequestedOutput(output_name, class_count=FLAGS.classes)
    # ]

    outputs = [
        client.InferRequestedOutput("detection_boxes"), #, class_count=FLAGS.classes),
        client.InferRequestedOutput("detection_scores") #, class_count=FLAGS.classes)
    ]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


def get_latest(Q):
    """Getting latest element from queue"""
    element = None
    while not Q.empty():
        element = Q.get(block=False)
    return element


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use streaming inference API. ' +
                        'The flag is only available with gRPC protocol.')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument(
        '-x',
        '--model-version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument(
        '-s',
        '--scaling',
        type=str,
        choices=['NONE', 'INCEPTION', 'VGG'],
        required=False,
        default='NONE',
        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='HTTP',
                        help='Protocol (HTTP/gRPC) used to communicate with ' +
                        'the inference service. Default is HTTP.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    if FLAGS.streaming and FLAGS.protocol.lower() != "grpc":
        raise Exception("Streaming is only allowed with gRPC protocol")

    try:
        if FLAGS.protocol.lower() == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
        else:
            # Specify large enough concurrency to handle the
            # the number of requests.
            concurrency = 20 if FLAGS.async_set else 1
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
        print("Model Metadata:", model_metadata)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version)
        print("Model Config:", model_config)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if FLAGS.protocol.lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(
            model_metadata, model_config)

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
        model_metadata, model_config)

    print("model parameters ", max_batch_size, input_name, output_name, c, h, w, format, dtype)

    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [
            os.path.join(FLAGS.image_filename, f)
            for f in os.listdir(FLAGS.image_filename)
            if os.path.isfile(os.path.join(FLAGS.image_filename, f))
        ]
        filenames = [name for name in filenames if name.endswith(".jpg")]
    else:
        filenames = [
            FLAGS.image_filename,
        ]

    filenames.sort()

    # Preprocess the images into input data according to model
    # requirements
    image_data = []
    for filename in filenames:
        img = Image.open(filename)
        image_data.append(
            preprocess(img, format, dtype, c, h, w, FLAGS.scaling,
                       FLAGS.protocol.lower()))

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    user_data = UserData()

    # Holds the handles to the ongoing HTTP async requests.
    async_requests = []

    sent_count = 0
    rec_count = 0

    if FLAGS.streaming:
        triton_client.start_stream(partial(completion_callback, user_data))

    print(gstreamer_pipeline(flip_method=2))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        # only works with batch_size 1 from here
        frame_id = 0
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            repeated_image_data = []

            # for idx in range(FLAGS.batch_size):
            #     repeated_image_data.append(image_data[image_idx])
            #     image_idx = (image_idx + 1) % len(image_data)
            #     if image_idx == 0:
            #         last_request = True
            #
            # if max_batch_size > 0:
            #     batched_image_data = np.stack(repeated_image_data, axis=0)
            # else:
            #     batched_image_data = repeated_image_data[0]
            t_cap = datetime.datetime.now()
            ret_val, img = cap.read()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize
            img = cv2.resize(img, (320, 320))

            # type casting
            # img = np.float32(img)

            # connect with old implementation
            batched_image_data = np.array([img], dtype="float32")

            t0 = datetime.datetime.now()
            # Send request
            try:
                if sent_count <= rec_count + 5:
                    for inputs, outputs, model_name, model_version in requestGenerator(
                            batched_image_data, input_name, output_name, dtype, FLAGS):
                        sent_count += 1
                        if FLAGS.streaming:
                            print(f"Sending request {sent_count}")
                            triton_client.async_stream_infer(
                                FLAGS.model_name,
                                inputs,
                                request_id=str(sent_count),
                                model_version=FLAGS.model_version,
                                outputs=outputs)
                        elif FLAGS.async_set:
                            if FLAGS.protocol.lower() == "grpc":
                                triton_client.async_infer(
                                    FLAGS.model_name,
                                    inputs,
                                    partial(completion_callback, user_data),
                                    request_id=str(sent_count),
                                    model_version=FLAGS.model_version,
                                    outputs=outputs)
                            else:
                                async_requests.append(
                                    triton_client.async_infer(
                                        FLAGS.model_name,
                                        inputs,
                                        request_id=str(sent_count),
                                        model_version=FLAGS.model_version,
                                        outputs=outputs))
                        else:
                            responses.append(
                                triton_client.infer(FLAGS.model_name,
                                                    inputs,
                                                    request_id=str(sent_count),
                                                    model_version=FLAGS.model_version,
                                                    outputs=outputs))

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if FLAGS.streaming:
                    triton_client.stop_stream()
                sys.exit(1)

            t1 = datetime.datetime.now()
            print(f"inf time: {(t1 - t0).total_seconds()}")
            #print(responses[-1].get_response().id)
            try:
                print("qsize", user_data._completed_requests.qsize())
                #(results, error) = user_data._completed_requests.get(block=False)
                results = get_latest(user_data._completed_requests)
                t2 = datetime.datetime.now()
                print(f"ret time: {(t2 - t1).total_seconds()}")
                print("Retrieved response with id: ", results.get_response().id)
                rec_count += 1
            except Exception as e:
                results = None
                print(e)
                break
                t2 = datetime.datetime.now()

            # try:
            if results is not None:
                img_annotated = visualize_detections_live_triton(img, results.as_numpy("detection_boxes"), results.as_numpy("detection_scores"), min_score=0.25)
            else:
                img_annotated = img
            # except Exception as e:
            #     print(e)
            #     img_annotated = img
            cv2.imshow("CSI Camera", np.asarray(img_annotated))
            t3 = datetime.datetime.now()
            print(f"show time: {(t3 - t2).total_seconds()}")

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                # ESC pressed
                print("'q' was hit, closing...")
                break

            print(f"framerate: {1/(t3 - t_cap).total_seconds()}")



    cap.release()
    cv2.destroyAllWindows()

    if FLAGS.streaming:
        triton_client.stop_stream()

    # if FLAGS.protocol.lower() == "grpc":
    #     if FLAGS.streaming or FLAGS.async_set:
    #         processed_count = 0
    #         while processed_count < sent_count:
    #             (results, error) = user_data._completed_requests.get()
    #             processed_count += 1
    #             if error is not None:
    #                 print("inference failed: " + str(error))
    #                 sys.exit(1)
    #             responses.append(results)
    # else:
    #     if FLAGS.async_set:
    #         # Collect results from the ongoing async requests
    #         # for HTTP Async requests.
    #         for async_request in async_requests:
    #             responses.append(async_request.get_result())
    #
    # for response in responses:
    #     # print(response)
    #     if FLAGS.protocol.lower() == "grpc":
    #         this_id = response.get_response().id
    #     else:
    #         this_id = response.get_response()["id"]
    #     # print("Request {}, batch size {}".format(this_id, FLAGS.batch_size))
    #     print(response.get_response().id)
    #     # # postprocess(response, output_name, FLAGS.batch_size, max_batch_size > 0)
    #     # print(response.as_numpy("detection_scores"))
    #     # print(response.as_numpy("detection_boxes"))

    print("PASS")
