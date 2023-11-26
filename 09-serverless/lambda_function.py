import tflite_runtime.interpreter as tflite

import os
from io import BytesIO
from urllib import request
from PIL import Image

import numpy as np

MODEL_NAME = os.getenv('MODEL_NAME', 'bees-wasps-v2.tflite')

def download_and_prepare_image(url, target_size):
    with request.urlopen(url) as resp:
        img = Image.open(BytesIO(resp.read()))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_resized = img.resize(target_size, Image.NEAREST)

    return img_resized

def prepare_input(x):
    x /= 255.0
    return x


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url):
    img = download_and_prepare_image(url, target_size=(150, 150))

    x = np.array(img, dtype='float32')

    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    return float(preds[0, 0])

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {'prediction': pred}

    return result