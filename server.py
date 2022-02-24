import os
import json
import logging
import numpy as np
from PIL import Image
from Yolov5.inference import run_inference
from flask import Flask, request, jsonify, abort

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
target_id = ""

@app.route("/health_check", methods=['GET', 'POST'])
def healthCheck():
    return jsonify(success=True)


@app.route("/obstacle", methods=['POST'])
def test_method():
    global target_id
    # print(request.json)
    # data from the connection
    if 'image' in request.files:
        photo = request.files['image']
        if photo.filename != '':
            photo.save(os.path.join(r'.', photo.filename))
            # process your img_arr here
            pil_image = Image.open(photo)
            # Run inference.py here. Returns output ID.
            target_id = str(run_inference(pil_image))
            print(type(target_id))
            print("from server", target_id)
    return target_id


def run_server_api():
    app.run(host='192.168.3.8', port=3001)


if __name__ == "__main__":
    run_server_api()