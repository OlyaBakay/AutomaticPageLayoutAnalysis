import copy
import json
import os
import numpy as np

class Region:
    TEXT = "text"
    IMAGE = "image"
    SEPARATOR = "separator"
    GRAPHIC = "graphic"

    PAGE_NUMBER = 'page-number'
    SIGNATURE_MARK = 'signature-mark'
    PARAGRAPH = 'paragraph'
    HEADER = 'header'
    CAPTION = 'caption'
    FOOTNOTE = 'footnote'
    HEADING = 'heading'
    MATHS = 'maths'

    HANDWRITTEN_ANNOTATION = 'handwritten-annotation'
    OTHER = 'other'
    SUBCATEGORIES = {TEXT: [PAGE_NUMBER, SIGNATURE_MARK, PARAGRAPH, HEADER, CAPTION, FOOTNOTE, HEADING],
                     IMAGE: [IMAGE],
                     SEPARATOR: [SEPARATOR],
                     MATHS: [MATHS],
                     GRAPHIC: [HANDWRITTEN_ANNOTATION, OTHER]}
    CATEGORIES = list(SUBCATEGORIES.keys())

    LEVEL_ALL = -1
    LEVEL_CATEGORY = 0
    LEVEL_SUBCATEGORY = 1

    def __init__(self, category="", subcategory="", contour=np.array([])):
        self.__category = category
        self.__subcategory = subcategory
        self.__category_id = self.__find_category_id()
        self.__subcategory_id = self.__find_subcategory_id()
        self.__contour = contour

    def transform(self, f=lambda x: x):
        return Region(self.__category, self.__subcategory, f(self.__contour))

    def __find_subcategory_id(self):
        id = 1
        for category in Region.CATEGORIES:
            if category == self.__category:
                return id + Region.SUBCATEGORIES[category].index(self.__subcategory)
            id += len(Region.SUBCATEGORIES[category])
        return -1

    def __find_category_id(self):
        return Region.CATEGORIES.index(self.__category) + 1

    @property
    def category(self):
        return self.__category

    @property
    def subcategory(self):
        return self.__category + '/' + self.__subcategory

    @property
    def category_id(self):
        return self.__category_id

    @property
    def subcategory_id(self):
        return self.__subcategory_id

    @property
    def contour(self):
        return self.__contour

def get_spaced_colors(n):
    max_value = 16581375
    interval = int(max_value / n)
    colors = [hex(i)[2:].zfill(6) for i in range(0, max_value, interval)]
    return [{"tuple": (int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)), "hex": i} for i in colors]

def generate_label_map(outpath, level=1):
    template = "item {{\n    id: {0}\n    name: '{1}'\n}}"
    items = []

    if level == Region.LEVEL_ALL:
        items.append(template.format(1, "all"))
    elif level == Region.LEVEL_CATEGORY:
        for i, category in enumerate(Region.CATEGORIES):
            items.append(template.format(i + 1, category))
    elif level == Region.LEVEL_SUBCATEGORY:
        i = 1
        for category in Region.CATEGORIES:
            for subcategory in Region.SUBCATEGORIES[category]:
                items.append(template.format(i, category + "/" + subcategory))
                i += 1

    with open(os.path.join(outpath, "label_map.pbtxt"), "w") as f:
        f.write("\n\n".join(items))



if __name__ == "__main__":
    generate_meta_json("../out/", level=1)