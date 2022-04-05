import numpy as np
import cv2
import tritonclient.grpc as grpc_client
import tritonclient.http as httpclient


MODEL_NAME = 'thymio_model'
URL = 'localhost:8000/v2/models/'
TRITON_CLIENT = httpclient.InferenceServerClient(url=URL)
INPUT_NAME = "input_tensor:0"


def get_bytes_img_request(batch=1):
    img = np.random.randint(low=0, high=255, size=(320, 320, 3), dtype=np.uint8)
    #_, buffer = cv2.imencode('.jpg', img)
    #input_bytes = np.array([buffer.tobytes()])
    #input_bytes = np.tile(input_bytes, (batch, 1))
    batch = np.array([img], dtype="float32")
    bytes_input = grpc_client.InferInput(INPUT_NAME, batch, 'FP32')
    bytes_input.set_data_from_numpy(input_bytes)
    return [bytes_input]


def send_request(inputs):
    return TRITON_CLIENT.infer(model_name=MODEL_NAME, inputs=inputs)


#inputs = get_bytes_img_request(batch=1)
img = np.random.randint(low=0, high=255, size=(320, 320, 3), dtype=np.uint8).astype(np.float32)
batch = np.array([img], dtype="float32")
print(batch.shape)
inputs=[httpclient.InferInput(INPUT_NAME, batch.shape, "FP32")]
inputs[0].set_data_from_numpy(batch)
outputs = [httpclient.InferRequestedOutput("detection_scores")]
response = TRITON_CLIENT.infer(MODEL_NAME,inputs,request_id=str(1),outputs=outputs)
print(response)
#output = send_request(inputs)
#print(output.get_response())
