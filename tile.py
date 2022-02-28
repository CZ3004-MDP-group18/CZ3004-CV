import tkinter
from tkinter import *
import os
from PIL import Image, ImageTk
import glob
#from server import run_directory

# tile_directory = 'runs/t8/run1'
# tile_directory = run_directory
# root = Tk()

def generate_images(tile_directory):
    root = Tk()

    # get number of images in tile_directory
    num_captured_images = len([name for name in os.listdir(tile_directory)])

    raw_images = []
    for filename in glob.glob(tile_directory + '/*.jpg'):
        raw_image = Image.open(filename)
        raw_images.append(raw_image)

    # resize images
    resized_images = []
    for img in raw_images:
        resized_image = img.resize((320, 240), Image.ANTIALIAS)
        resized_images.append(resized_image)

    # get list of PhotoImages
    photo_list = []
    for img in resized_images:
        photo = ImageTk.PhotoImage(img)
        photo_list.append(photo)

    # display on grid
    nr = 3  # max number of rows
    nc = 3  # max number of columns

    labelled_list = []
    for i in range(num_captured_images):
        labelled_list.append(Label(root, image=photo_list[i]))
        labelled_list[-1].grid(row=i // nr, column=i % nc)
        print(i)

    root.mainloop()
