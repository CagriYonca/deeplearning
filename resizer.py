from PIL import Image, ImageFile
import os, sys, glob
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_dir = "/mnt/data/data/summer_2018/target_files/"
target_dir = "/mnt/data/data/summer_2018/resized_target_files/"

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

def resize(input_size):
    counter = 0
    for item in glob.glob(os.path.join(data_dir, "*.jpg")):
        item = os.path.basename(item)
        counter += 1
        actual_file_path = os.path.join(data_dir, item)
        if os.path.isfile(os.path.join(target_dir, item)):
            continue
        im = Image.open(actual_file_path)
        width, height = im.size
        if height > width:
            new_h, new_w = input_size * height / width, input_size
        else:
            new_h, new_w = input_size, input_size * width / height
        filename = os.path.split(actual_file_path)[-1]
        print(counter)
        imResize = im.resize((int(new_h), int(new_w)), Image.ANTIALIAS)
        imResize.save(os.path.join(target_dir, filename), "JPEG", quality=90)

resize(256)
