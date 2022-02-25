"""
This file runs inference on input images stored in 'captured'
and outputs labelled images in 'run'
Returns image IDs.
"""

import torch
import requests
import subprocess
from PIL import Image
import glob
import pandas
import numpy as np

# dictionary of image IDs. Keys are IDs, values are bounding box labels
image_ids = {'square' : 0,
            'blue1' : 11,
            'green2' : 12,
            'red3' : 13,
            'white4' : 14,
            'yellow5' : 15,
            'blue6' : 16,
            'green7' : 17,
            'red8' : 18,
            'white9' : 19,
            'red-a' : 20,
            'green-b' : 21,
            'white-c' : 22,
            'blue-d' : 23,
            'yellow-e' : 24,
            'red-f' : 25,
            'green-g' : 26,
            'white-h' : 27,
            'blue-s' : 28,
            'yellow-t' : 29,
            'red-u' : 30,
            'green-v' : 31,
            'white-w' : 32,
            'blue-x': 33,
            'yellow-y': 34,
            'red-z': 35,
            'white-up' : 36,
            'red-down': 37,
            'green-right': 38,
            'blue-left' : 39,
            'yellow-circle' : 40
            }

# run inference. Get labelled images
# print("about to run command shell")
# subprocess.run(['python detection/yolov5/detect.py --weights detection/yolov5/models/best_14feb.pt --img 416 --conf 0.1 --source captured/close10.jpg'],
#                shell=True)
# print("ran command shell")

# retrieve images from captured. Only for testing
input_images = []
for filename in glob.glob('captured/*.jpg'): #assuming jpg
    im=Image.open(filename)
    input_images.append(im)

model = torch.hub.load(r"C:\Users\okapu\Desktop\1Uni\AY2021-22 Sem 2\3004 MDP\CZ3004-CV\pretrained\ultralytics_yolov5_master", 'custom',
                        r'C:\Users\okapu\Desktop\1Uni\AY2021-22 Sem 2\3004 MDP\CZ3004-CV\Yolov5\detection\yolov5\models\best_14feb.pt', source="local")


def run_inference(image):
    # try 'ultralytics/yolov5' as 1st argument if cannot run
    result = model(image)
    result.print()
    result_df = result.pandas().xyxy[0]
    classes = result_df['name'].values.tolist()
    if len(classes) == 0:
        output_id = 99
        return [99,0]
    elif len(classes) >= 2:
        max_confidence = max(result_df['confidence'].values.tolist())
        if max_confidence >= 0.8:
            class_name = classes[0]
            output_id = image_ids['%s' % class_name]
        else:
            output_id = 99
        max_class = result_df.loc[result_df['confidence'] == max_confidence, 'name'].iloc[0]
        output_id = image_ids['%s' % max_class]
    elif len(classes) == 1:
        max_confidence = max(result_df['confidence'].values.tolist())
        print("max confidence", max_confidence)
        if max_confidence >= 0.8:
            class_name = classes[0]
            output_id = image_ids['%s' % class_name]
        else:
            output_id = 99
    print("from inference", output_id)
    result.save()

    # get output distance
    ymax = result_df['ymax'][0]
    ymin = result_df['ymin'][0]
    angle = 9
    distance = calculate_distance(ymin, ymax, angle)

    outputs = [output_id, distance]
    return outputs

def calculate_distance(ymin, ymax, angle):
    oppcm = ((ymax - ymin) * 0.0264583)/2
    while True:
        if oppcm >= 0:
            angle = oppcm*(np.log(np.power(10,(oppcm*0.58))))
        return (oppcm/np.tan(angle*0.0175))

# --- For Testing ---
for i in input_images:
    # try 'ultralytics/yolov5' as 1st argument if cannot run
    result = model(i)
    result.print()
    result_df = result.pandas().xyxy[0]
    print(result_df)
    classes = result_df['name'].values.tolist()
    if len(classes) == 0:
        output_id = 99
    elif len(classes) >= 2:
        max_confidence = max(result_df['confidence'].values.tolist())
        print("max confidence", max_confidence)
        if max_confidence >= 0.8:
            class_name = classes[0]
            output_id = image_ids['%s' % class_name]
        else:
            output_id = 99
            pass
        max_class = result_df.loc[result_df['confidence'] == max_confidence, 'name'].iloc[0]
        output_id = image_ids['%s' % max_class]
    elif len(classes) == 1:
        max_confidence = max(result_df['confidence'].values.tolist())
        print("max confidence", max_confidence)
        if max_confidence >= 0.8:
            class_name = classes[0]
            output_id = image_ids['%s' % class_name]
        else:
            output_id = 99


    ymax = result_df['ymax'][0]
    ymin = result_df['ymin'][0]
    angle = 9
    height = ymax - ymin
    print("height", (height* 0.0264583))
    distance = calculate_distance(ymin, ymax, angle)
    outputs = (output_id, distance)
    print(outputs)