import logging
import os

from PIL import Image
from flask import Flask, request, jsonify

from Yolov5.inference import run_inference

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
target_id = ''
distance = ''


@app.route("/health_check", methods=['GET', 'POST'])
def healthCheck():
    return jsonify(success=True)


@app.route("/obstacle", methods=['POST'])
def test_method():
    global distance
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
            outputs = run_inference(pil_image)
            target_id = str(outputs[0])
            distance = str(outputs[1])
            if target_id == '99':
                target_id = ""

    output = target_id +","+distance
    print(type(output))
    print("from server", output)
    return output


def run_server_api():
    app.run(host='192.168.3.13', port=4000)


if __name__ == "__main__":
    run_server_api()
