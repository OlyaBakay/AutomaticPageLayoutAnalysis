import os
from pathlib import Path
from xml.dom.minidom import parse
import cv2
import numpy as np
from image import Image, Region
import tensorflow as tf
from tqdm import tqdm

from utils.region import generate_label_map


def find_regions(dom, tag_class):
    regions = []
    region_tags = dom.getElementsByTagName(tag_class.capitalize() + "Region")

    for region_tag in region_tags:
        if region_tag.hasAttribute("type"):
            region_type = region_tag.getAttribute("type")
        else:
            region_type = tag_class

        point_tags = region_tag.getElementsByTagName("Point")
        countour = []
        for point_tag in point_tags:
            countour.append((point_tag.getAttribute("x"), point_tag.getAttribute("y")))
        regions.append(Region(tag_class, region_type, np.array(countour, int)))
    return regions

def parse_xml(file):
    file = Path(file)
    dom = parse(str(file))
    image_filename = dom.getElementsByTagName("Page")[0].getAttribute('imageFilename')
    image_path = str(file.parent / image_filename)
    image = cv2.imread(image_path)

    image_object = Image(image, filename=image_filename)
    image_object.regions += find_regions(dom, Region.TEXT)
    image_object.regions += find_regions(dom, Region.IMAGE)
    image_object.regions += find_regions(dom, Region.SEPARATOR)
    image_object.regions += find_regions(dom, Region.GRAPHIC)
    return image_object

def to_tfrecords(in_path, out_path, level=Region.LEVEL_CATEGORY):
    files = list(filter(lambda x: x.endswith(".xml"), os.listdir(in_path)))
    path = Path(in_path)
    generate_label_map(out_path, level=level)
    writer = tf.python_io.TFRecordWriter(os.path.join(out_path, "src.record"))
    for file in tqdm(files):
        file = path / file
        image_object = parse_xml(file)
        image_object = image_object.correct()
        writer.write(image_object.to_tfrecord(level=level))
    writer.close()

if __name__ == "__main__":
    # images = to_tfrecords("../src/impact", "../out/", level=Region.LEVEL_SUBCATEGORY)
    # ../data/impact/pc-00539981.xml
    #for file in os.listdir("../data/impact"):
        # if file.endswith(".xml"):
    image = parse_xml("../data/impact/pc-00539980.xml")
        #     cv2.imwrite(os.path.join("../out/data", image.filename + ""), image.draw_regions())

    #image = cv2.imread("../data/supervisely/zbirnyk/tom_1/1108-2162-1-PB/img/img_0.jpg")
    #image = Image(image)
    # image = image.correct()
    # print(image.image)
    cv2.imwrite("../out/out.jpg", image.draw_regions(level=Region.LEVEL_SUBCATEGORY))
    # print(info)
