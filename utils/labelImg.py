import os
from pathlib import Path

import cv2
#import tensorflow as tf
from tqdm import tqdm
from image import Image
from xml.dom.minidom import parse
import pandas as pd

def statistic(inpath):
    info = {}
    for path in os.listdir(inpath):
        if path.endswith(".xml"):
            dom = parse(os.path.join(inpath, path))
            regions = dom.getElementsByTagName("object")
            for region in regions:
                region_type = region.getElementsByTagName("name").item(0).firstChild.nodeValue
                info[region_type] = info.get(region_type, 0) + 1
    return pd.Series(info)

if __name__ == "__main__":
    sr = statistic("../data/labelImg/part_1")
    print(sr.sort_values())