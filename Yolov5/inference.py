"""
This file runs inference on input images stored in 'captured' (testing) or run
and outputs labelled images in run directory
Returns image IDs and distance
"""

import torch
import requests
import subprocess
from PIL import Image
import glob
import pandas
import numpy as np

output_id = 0
distance = 0.0
confidence_level = 0.05

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

# model = torch.hub.load(r"C:\Users\okapu\Desktop\1Uni\AY2021-22 Sem 2\3004 MDP\CZ3004-CV\pretrained\ultralytics_yolov5_master", 'custom',
#                         r'C:\Users\okapu\Desktop\1Uni\AY2021-22 Sem 2\3004 MDP\CZ3004-CV\Yolov5\detection\yolov5\models\best_14feb.pt', source="local")

model = torch.hub.load(r"pretrained\ultralytics_yolov5_master", 'custom',
                        r"Yolov5\detection\yolov5\models\best_14feb.pt", source="local")


# output ID meanings
# 0 - bullseye
# 99 - blank
# others refer to target_ID of character images

# pass in 2 parameters - image + run directory
def run_inference(image, run_directory):
    global output_id
    global distance

    print("\n")

    record_directory = 'runs/w9_testing'  # directory to save all possible images without bounding box. for testing purposes.

    # try 'ultralytics/yolov5' as 1st argument if cannot run
    result = model(image)
    #result.print()
    result_df = result.pandas().xyxy[0]
    result_df = result_df.sort_values(by=['confidence'], ascending=False).reset_index(drop=True)
    print("======= NEW IMAGE ======================================================================")
    print(result_df)
    classes = result_df['name'].values.tolist()
    result.save(best_class="ignore", save_dir=record_directory, is_testing=True)

    # --- Week 9 ---
    # if len(classes) == 0:
    #     print("=== 0 CLASSES IDENTIFIED ===")
    #     return [99, 0]
    # elif len(classes) == 1:
    #     print("=== ONE CLASS IDENTIFIED ===")
    #     confidence = result_df['confidence'].iloc[0]
    #     if confidence >= confidence_level:
    #         class_name = result_df.loc[result_df['confidence'] == confidence, 'name'].iloc[0]
    #         result.save(best_class=class_name, save_dir=run_directory)
    #         output_id = image_ids['%s' % class_name]
    #         ymax = result_df['ymax'][0]
    #         ymin = result_df['ymin'][0]
    #         angle = 9
    #         distance = calculate_distance(ymin, ymax, angle)
    #         print("distance is", distance)
    #         if output_id == 0:
    #             return [0, distance]
    #         else:
    #             print("no bullseye detected")
    #             return [99, 0]
    # elif len(classes) == 2:
    #     print("=== 2 CLASSES IDENTIFIED ===")
    #     # check that confidence levels are above threshold
    #     confidence_one = max(result_df['confidence'].values.tolist())
    #     confidence_two = min(result_df['confidence'].values.tolist())
    #     print("confidence one and two are " + str(confidence_one) + "," + str(confidence_two))
    #     if confidence_one >= confidence_level and confidence_two >= confidence_level:
    #         class_one = result_df.loc[result_df['confidence'] == confidence_one, 'name'].iloc[0]
    #         class_two = result_df.loc[result_df['confidence'] == confidence_two, 'name'].iloc[0]
    #         output_id_one = image_ids['%s' % class_one]
    #         output_id_two = image_ids['%s' % class_two]
    #         result.save(best_class=class_one, save_dir=run_directory)
    #         result.save(best_class=class_two, save_dir=run_directory)
    #
    #         distances = []
    #         xmax_list = []
    #         xmin_list = []
    #         for i in range(len(classes)):
    #             row_ymax = result_df['ymax'][i]
    #             row_ymin = result_df['ymin'][i]
    #             row_xmax = result_df['xmax'][i]
    #             row_xmin = result_df['xmin'][i]
    #             angle = 9
    #             xmax_list.append(row_xmax)
    #             xmin_list.append(row_xmin)
    #             row_distance = calculate_distance(row_ymin, row_ymax, angle)
    #             distances.append(row_distance)
    #         print("from inference. distances is", distances)  # DO NOT SORT DISTANCES HERE. RETAIN INDEXES
    #         # print("xmax_list", xmax_list)
    #         # print("xmin_list", xmin_list)
    #         # average_x = (min(xmin_list) + max(xmax_list)) / 2
    #         result_df['distance'] = distances
    #         print(result_df)
    #         # check both classes are bullseye
    #         if output_id_one == 0 and output_id_two == 0:
    #             min_dist = min(distances[0], distances[1])
    #             print("minimum distance between 2 targets", min_dist)
    #             return [0, min_dist]
    #         else: # both weren't bullseye
    #             print("both weren't bullseye")
    #             return [99, 0]
    #     else: # both weren't above threshold level
    #         return [99, 0]
    # elif len(classes) > 2:
    #     print("=== MORE THAN 2 CLASSES IDENTIFIED ===")
    #     # get 2 classes with highest confidence level
    #     confidence_one = result_df['confidence'].iloc[0]
    #     confidence_two = result_df['confidence'].iloc[1]
    #     print("confidence one and two are " + str(confidence_one) + "," + str(confidence_two))
    #     if confidence_one >= confidence_level and confidence_two >= confidence_level:
    #         class_one = result_df.loc[result_df['confidence'] == confidence_one, 'name'].iloc[0]
    #         class_two = result_df.loc[result_df['confidence'] == confidence_two, 'name'].iloc[0]
    #         result.save(best_class=class_one, save_dir=run_directory)
    #         result.save(best_class=class_two, save_dir=run_directory)
    #         output_id_one = image_ids['%s' % class_one]
    #         output_id_two = image_ids['%s' % class_two]
    #         distances = []
    #         for i in range(len(classes)):
    #             row_ymax = result_df['ymax'][i]
    #             row_ymin = result_df['ymin'][i]
    #             angle = 9
    #             row_distance = calculate_distance(row_ymin, row_ymax, angle)
    #             distances.append(row_distance)
    #         print("from inference. distances is", distances)  # DO NOT SORT DISTANCES HERE. RETAIN INDEXES
    #         result_df['distance'] = distances
    #         print(result_df)
    #         if output_id_one == 0 and output_id_two == 0:
    #             min_dist = min(distances[0], distances[1])
    #             print("minimum distance between 2 targets", min_dist)
    #             return [0, min_dist]
    #         else: # both weren't bullseye
    #             print("the two highest confidence images weren't bullseye. iterating thru all detections...")
    #             # identify all classes with bullseye (may identify 1 of the 2 above)
    #             bullseye_classes = [] #stores position (i) of all detections identified as bullseye
    #             distances = []
    #             for i in range(len(classes)):
    #                 confidence_i = result_df['confidence'].iloc[i]
    #                 class_i = result_df.loc[result_df['confidence'] == confidence_i, 'name'].iloc[0]
    #                 output_id_i = image_ids['%s' % class_i]
    #                 print("current output id", output_id_i)
    #                 if output_id_i == 0:
    #                     bullseye_classes.append(i)
    #                 i += 1
    #             for j in range(len(bullseye_classes)):
    #                 row_ymax = result_df['ymax'][bullseye_classes[j]]
    #                 row_ymin = result_df['ymin'][bullseye_classes[j]]
    #                 angle = 9
    #                 row_distance = calculate_distance(row_ymin, row_ymax, angle)
    #                 distances.append(row_distance)
    #                 j += 1
    #             # check if there's any other bullseye identified
    #             if len(bullseye_classes) < 2:
    #                 print("less than 2 bullseyes detected")
    #                 return [99, 0]
    #             elif len(bullseye_classes) >= 2:
    #                 # only consider closest 2 images
    #                 distances.sort()
    #                 min_dist = min(distances[0], distances[1])
    #                 print("minimum distance between 2 targets", min_dist)
    #                 return [0, min_dist]
    #     else: # classes with highest confidence levels weren't above confidence level
    #         return [99, 0]
    #
    # # send outputs to server
    # print("from inference. output id is", output_id)
    # print("from inference. distance is", distance)
    # outputs = [output_id, distance]
    # print("from inference. outputs are", outputs)
    # return outputs

    # --- Week 8 ---
    if len(classes) == 0:
        print("=== 0 CLASSES IDENTIFIED ===")
        output_id = 99
        distance = 0
        # only for debugging to see pic
        result.save(best_class="blank", save_dir=run_directory) #non-target
        # return [99,0]
    elif len(classes) >= 2:
        print("=== MULTIPLE CLASSES IDENTIFIED ===")
        print("no. of classes", len(classes))
        max_confidence = max(result_df['confidence'].values.tolist())
        print("max confidence", max_confidence)
        if max_confidence >= confidence_level:
            max_class = result_df.loc[result_df['confidence'] == max_confidence, 'name'].iloc[0]
            print("max class",max_class)

            # get distances of each class identified
            distances = []
            for i in range(len(classes)):
                row_ymax = result_df['ymax'][i]
                row_ymin = result_df['ymin'][i]
                angle = 9
                row_distance = calculate_distance(row_ymin, row_ymax, angle)
                distances.append(row_distance)
            print("from inference. distances is", distances) # DO NOT SORT DISTANCES HERE. RETAIN INDEXES
            result_df['distance'] = distances
            print(result_df)

            # check if class with highest confidence is shortest distance
            short_distance = min(distances)
            max_conf_distance = distances[0]
            print("distance of highest confidence is", max_conf_distance)

            if max_conf_distance == short_distance:
                # get output id
                output_id = image_ids['%s' % max_class]
                # only save results that have non-blank/non-bullseye class ID
                if output_id == 0 or output_id == 99:
                    # pass
                    result.save(best_class="ignore", save_dir=run_directory) #non-target
                else:
                    result.save(best_class=max_class, save_dir=run_directory)
            else: # max confidence class is not closest to camera. Check if closest image is above threshold
                print("class with highest confidence is not closest!")
                # get confidence of class that's shortest distance to camera
                short_confidence = result_df.loc[result_df['distance'] == short_distance, 'confidence'].iloc[0]
                print("short confidence", short_confidence)
                if short_confidence >= confidence_level:
                    short_class = result_df.loc[result_df['confidence'] == short_confidence, 'name'].iloc[0]
                    print("short class", short_class)
                    output_id = image_ids['%s' % short_class]

                    # only save results that have non-blank/non-bullseye class ID
                    if output_id == 0 or output_id == 99:
                        # pass
                        result.save(best_class="ignore", save_dir=run_directory) #non-target
                    else:
                        result.save(best_class=short_class, save_dir=run_directory)
                else: # closest image is not above threshold.
                    output_id = 99
                    # iterate through non-highest-confidence images further from closest
                    # by dropping shortest class and finding new shortest
                    for i in range(len(classes) - 1):
                        result_df.drop(result_df[result_df['distance'] == short_distance].index, inplace=True)
                        short_distance = min(result_df['distance'].values.tolist())
                        print("second shortest distance is", short_distance)
                        short_confidence = result_df.loc[result_df['distance'] == short_distance, 'confidence'].iloc[0]
                        if short_confidence >= confidence_level:
                            short_class = result_df.loc[result_df['confidence'] == short_confidence, 'name'].iloc[0]
                            print("second shortest class", short_class)
                            output_id = image_ids['%s' % short_class]
                            if output_id == 0 or output_id == 99:
                                # pass
                                result.save(best_class="ignore", save_dir=run_directory) #non-target
                            else:
                                result.save(best_class=short_class, save_dir=run_directory)
                                break
                        else:
                            result.save(best_class="ignore", save_dir=run_directory) #non-target
                            continue

            distance = short_distance

            # uncomment the following if not using distances
            # get output id
            # output_id = image_ids['%s' % max_class]
            # # only save results that have non-blank/non-bullseye class ID
            # if output_id == 0 or output_id == 99:
            #     pass
            # else:
            #     result.save(best_class=max_class, save_dir=run_directory)

        else:
            output_id = 99
            result.save(best_class="ignore", save_dir=run_directory) #non-target
    elif len(classes) == 1:
        print("=== ONE CLASS IDENTIFIED ===")
        max_confidence = max(result_df['confidence'].values.tolist())
        print("max confidence", max_confidence)
        ymax = result_df['ymax'][0]
        ymin = result_df['ymin'][0]
        angle = 9
        distance = calculate_distance(ymin, ymax, angle)
        print("from inference. ymax is", ymax)
        print("from inference. ymin is", ymin)
        print("from inference. distance is", distance)

        if max_confidence >= confidence_level:
            class_name = classes[0]
            output_id = image_ids['%s' % class_name]
            # only save results that have non-blank/non-bullseye class ID
            if output_id == 0 or output_id == 99:
                result.save(best_class="ignore", save_dir=run_directory) #non-target
            else:
                result.save(best_class=class_name,save_dir=run_directory)
        else:
            output_id = 99
            result.save(best_class="ignore", save_dir=run_directory) #non-target
    # print("from inference. output id is", output_id)
    # # result.save() # only save results that have non-blank/non-bullseye class ID
    #
    # # uncomment the following if not using distances
    # # get output distance of target image
    # # ymax = result_df['ymax'][0]
    # # ymin = result_df['ymin'][0]
    # # angle = 9
    # # distance = calculate_distance(ymin, ymax, angle)
    # # print("from inference. ymax is", ymax)
    # # print("from inference. ymin is", ymin)
    # print("from inference. distance is", distance)
    #
    # outputs = [output_id, distance]
    # print("from inference. outputs are", outputs)
    # return outputs

def calculate_distance(ymin, ymax, angle):
    oppcm = ((ymax - ymin) * 0.0264583)/2
    while True:
        if oppcm >= 0:
            angle = oppcm*(np.log(np.power(10,(oppcm*0.675))))
        return (oppcm/np.tan(angle*0.0175))

# --- For Testing --- Saved images go to captured
# retrieve images from captured. Only for testing
# input_images = []
# for filename in glob.glob('captured/*.jpg'): #assuming jpg
#     im=Image.open(filename)
#     input_images.append(im)
#     run_inference(im, run_directory='runs/test_output')