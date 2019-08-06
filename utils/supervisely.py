import json
import os
import pickle
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.image import Image
from utils.region import Region, get_spaced_colors, generate_label_map

ANNOTATION_FOLDER = "ann"
IMAGE_FOLDER = "img"


def generate_meta_json(outpath, level=1):
    info = {"classes": [], "tags_images": [], "tags_objects": []}

    if level == Region.LEVEL_CATEGORY:
        amount = len(Region.CATEGORIES)
        colors = get_spaced_colors(amount)
        for i, category in enumerate(Region.CATEGORIES):
            class_info = {"title": category, "shape": "rectangle",
                          "color": "#{}".format(colors[i]["hex"]), "geometry_settings": {}}
            info["classes"].append(class_info)
    elif level == Region.LEVEL_SUBCATEGORY:
        i = 1
        amount = sum([len(subcategories) for subcategories in Region.SUBCATEGORIES.values()])
        colors = get_spaced_colors(amount)

        for category in Region.CATEGORIES:
            for subcategory in Region.SUBCATEGORIES[category]:
                class_info = {"title": category + "/" + subcategory, "shape": "rectangle",
                              "color": "#{}".format(colors[i]["hex"]), "geometry_settings": {}}
                info["classes"].append(class_info)
                i += 1

    with open(os.path.join(outpath + "meta.json"), "w") as f:
        json.dump(info, f)


def prepare_dataset(inpath):
    files = [os.path.join(inpath, file) for file in os.listdir(inpath)]
    os.makedirs(os.path.join(inpath, ANNOTATION_FOLDER))
    os.makedirs(os.path.join(inpath, IMAGE_FOLDER))
    for file in files:
        filename, _ = os.path.splitext(os.path.basename(file))
        img = cv2.imread(file)
        h, w = img.shape[0], img.shape[1]
        info = {"tags": [], "description": "", "objects": [], "size": {"height": h, "width": w}}
        with open(os.path.join(inpath, ANNOTATION_FOLDER, "{}.jpg.json".format(filename)), "w") as f:
            json.dump(info, f)
        shutil.move(file, os.path.join(inpath, IMAGE_FOLDER, "{}.jpg".format(filename)))


def find_regions(info):
    objects = info['objects']
    regions = []
    for region in objects:
        class_splited = region['classTitle'].split("/")
        category = class_splited[0]
        subcategory = class_splited[1]
        countour = region['points']['exterior']
        regions.append(Region(category, subcategory, np.array(countour, int)))
    return regions


def parse_json(file):
    _dirname = os.path.dirname(os.path.dirname(file))
    basename = os.path.basename(file)
    image_filename = os.path.splitext(basename)[0]
    image_path = os.path.join(_dirname, IMAGE_FOLDER, image_filename)

    image = cv2.imread(image_path)
    image_object = Image(image, filename=image_filename)

    with open(file) as f:
        info = json.load(f)
        image_object.regions = find_regions(info)
    return image_object


def to_tfrecords(in_path, out_path, level=Region.LEVEL_CATEGORY, filename="data.record"):
    os.makedirs(out_path, exist_ok=True)
    files = os.listdir(os.path.join(in_path, ANNOTATION_FOLDER))
    generate_label_map(out_path, level=level)
    writer = tf.python_io.TFRecordWriter(os.path.join(out_path, filename))
    for file in tqdm(files):
        file = os.path.join(in_path, ANNOTATION_FOLDER, file)
        image_object = parse_json(file)
        image_object = image_object.correct()
        writer.write(image_object.to_tfrecord(level=level))
    writer.close()


def extract_projections(in_path, out_path, categories=("text", "maths", "separator")):
    # TODO: check it out
    for category in categories:
        os.makedirs(os.path.join(out_path, category), exist_ok=True)
    files = os.listdir(os.path.join(in_path, ANNOTATION_FOLDER))
    counters = {category: 0 for category in categories}
    for file in tqdm(files):
        file = os.path.join(in_path, ANNOTATION_FOLDER, file)
        image_object = parse_json(file)
        grey = cv2.cvtColor(image_object.image, cv2.COLOR_BGR2GRAY)

        for region in image_object.regions:
            if region.category in categories:
                mn = np.min(region.contour, axis=0)
                mx = np.max(region.contour, axis=0)
                block = grey[mn[1]:mx[1], mn[0]:mx[0]] / 255
                block = cv2.resize(block, (100, 100))
                x = block.sum(axis=0) / 100
                y = block.sum(axis=1) / 100
                info = {"x": np.array(x, np.float32), "y": np.array(y, np.float32)}
                info_path = os.path.join(out_path, region.category, "info_{}.pickle".format(counters[region.category]))
                with open(info_path, 'wb') as f:
                    pickle.dump(info, f)
                counters[region.category] += 1


if __name__ == "__main__":
    # extract_projections("../data/supervisely/zbirnyk/tom_1/1120-2163-1-PB", "../out/projections/1120-2163-1-PB")
    # extract_projections("../data/supervisely/zbirnyk/tom_1/1108-2162-1-PB", "../out/projections/1108-2162-1-PB")

    # cv2.waitKey(0)
    # to_tfrecords("../data/supervisely/zbirnyk/tom_1/1120-2163-1-PB", "../out/zbirnyk/subcategory",
    #              level=Region.LEVEL_SUBCATEGORY, filename="train.record")
    # to_tfrecords("../data/supervisely/zbirnyk/tom_1/1108-2162-1-PB", "../out/zbirnyk/subcategory",
    #              level=Region.LEVEL_SUBCATEGORY, filename="test.record")
    pass
