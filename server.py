import logging
import os
import glob
# import torchvision.transforms as T

from PIL import Image
from flask import Flask, request, jsonify

from Yolov5.inference import run_inference
from tile import generate_images

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
target_id = ''
distance = ''

# generates new run directory to save images in
run_directory = 'runs/t8/run1' # comment everything but this line for testing
run_number = 1
while os.path.exists(run_directory):
    run_number += 1
    run_directory = 'runs/t8/run' + ('%i' % run_number)
print("new run directory", run_directory)
os.makedirs(run_directory)

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
            print("from server.py photo.filename is", photo.filename)
            photo.save(os.path.join(r'.', photo.filename))
            # process your img_arr here
            pil_image = Image.open(photo)
            # Run inference.py here. Returns output ID.
            outputs = run_inference(pil_image, run_directory)
            target_id = str(outputs[0])
            distance = str(outputs[1])
            if target_id == '99':
                target_id = ""

    output = target_id +","+distance
    print(type(output))
    print("from server", output)

    # update number of images saved in run directory. Generate tile once complete
    num_captured_images = len([name for name in os.listdir(run_directory)])
    print("no of captured images so far: ", num_captured_images)
    if num_captured_images >= 4:
        generate_images(run_directory)

    return output

# --- For testing ---
# retrieve images from run_directory
test_directory = 'runs/t8/for_testing'
input_images = []
for filename in glob.glob(test_directory+'/*.jpg'): #assuming jpg
    pil_image=Image.open(filename)
    # pil_image = T.functional.adjust_sharpness(img=pil_image, sharpness_factor=3.0)
    # pil_image = T.functional.adjust_saturation(img=pil_image, saturation_factor=1.2)
    # pil_image = T.functional.adjust_contrast(img=pil_image, contrast_factor=1.5)
    # # pil_image = T.functional.adjust_gamma(img=pil_image, gamma=1.2)
    # pil_image = T.functional.adjust_brightness(img=pil_image, brightness_factor=1.5)

    input_images.append(pil_image)
    run_inference(pil_image, run_directory)
print("run directory passing to inference: ", run_directory)

num_captured_images = len([name for name in os.listdir(run_directory)])
print("no of captured images so far: ", num_captured_images)
if num_captured_images >= 4:
    generate_images(run_directory)

def run_server_api():
    app.run(host='192.168.3.12', port=4000)

if __name__ == "__main__":
    run_server_api()