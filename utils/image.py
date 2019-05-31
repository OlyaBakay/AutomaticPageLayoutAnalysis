import cv2
import numpy as np
import copy
import tensorflow as tf
import tfrecords
import os

from utils.region import Region


def _transform_regions(regions, transform=lambda x: x):
    new_regions = copy.deepcopy(regions)
    for region in new_regions:
        for i, contour in enumerate(new_regions[region]):
            new_regions[region][i] = transform(contour)
    return new_regions


class Image:
    def __init__(self, image, filename="image.jpg"):
        self.image = image
        self.filename = filename
        self.regions = []

    def scale(self, size):
        new_image = Image(cv2.resize(self.image, size))
        scale = size[0] / self.width,  size[1] / self.height
        f = lambda x: np.array(x * scale, int)
        new_image.regions = list(map(lambda x: x.transform(f), self.regions))
        return new_image

    def regions_to_rectangles(self):
        def f(x):
            x, y, w, h = cv2.boundingRect(x)
            return np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], int)

        new_image = Image(self.image.copy(), self.filename)
        new_image.regions = list(map(lambda x: x.transform(f), self.regions))
        return new_image

    def correct(self):
        image_grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        image_grey = cv2.GaussianBlur(image_grey, (1, 1), 0)
        ret, image_grey = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(image_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = list(map(cv2.contourArea, contours))
        largest_contour_idx = contour_areas.index(max(contour_areas))
        x, y, w, h = cv2.boundingRect(contours[largest_contour_idx])

        if w / self.width > 0.9 and h / self.height > 0.9:
            image_grey = image_grey[y:y + h,x:x + w]


        mask = image_grey != 255
        coords = np.argwhere(mask)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        image_grey = image_grey[x0:x1, y0:y1]

        shift = (x0 + x, y0 + y)
        w, h = image_grey.shape

        def f(x):
            x = x - shift
            x[:,0] = np.where(x[:,0] >= 0, x[:,0], 0)
            x[:,1] = np.where(x[:,1] >= 0, x[:,1], 0)
            x[:,0] = np.where(x[:,0] < h, x[:,0], h - 1)
            x[:,1] = np.where(x[:,1] < w, x[:,1], w - 1)
            return x

        new_image = Image(image_grey, self.filename)
        new_image.regions = list(map(lambda x: x.transform(f), self.regions))
        return new_image

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    def __create_labels(self, level):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for region in self.regions:
            xmins.append(region.contour[:,0].min() / self.width)
            xmaxs.append(region.contour[:,0].max() / self.width)
            ymins.append(region.contour[:,1].min() / self.height)
            ymaxs.append(region.contour[:,1].max() / self.height)
            if level == Region.LEVEL_ALL:
                class_name, class_id = "all", 1
            elif level == Region.LEVEL_CATEGORY:
                class_name, class_id = region.category, region.category_id
            elif level == Region.LEVEL_SUBCATEGORY:
                class_name, class_id = region.subcategory, region.subcategory_id

            classes_text.append(class_name.encode('utf8'))
            classes.append(class_id)

        return xmins, xmaxs, ymins, ymaxs, classes_text, classes

    def to_tfrecord(self, level=Region.LEVEL_CATEGORY):
        filename = self.filename.encode('utf8')
        encoded_jpg = cv2.imencode('.jpg', self.image)[1].tostring()
        image_format = b'jpg'
        xmins, xmaxs, ymins, ymaxs, classes_text, classes = self.__create_labels(level)
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tfrecords.int64_feature(self.height),
            'image/width': tfrecords.int64_feature(self.width),
            'image/filename': tfrecords.bytes_feature(filename),
            'image/source_id': tfrecords.bytes_feature(filename),
            'image/encoded': tfrecords.bytes_feature(encoded_jpg),
            'image/format': tfrecords.bytes_feature(image_format),
            'image/object/bbox/xmin': tfrecords.float_list_feature(xmins),
            'image/object/bbox/xmax': tfrecords.float_list_feature(xmaxs),
            'image/object/bbox/ymin': tfrecords.float_list_feature(ymins),
            'image/object/bbox/ymax': tfrecords.float_list_feature(ymaxs),
            'image/object/class/text': tfrecords.bytes_list_feature(classes_text),
            'image/object/class/label': tfrecords.int64_list_feature(classes),
        }))
        return tf_example.SerializeToString()


    def draw_regions(self, level=Region.LEVEL_CATEGORY):
        image_copy = self.image.copy()
        for i, region in enumerate(self.regions):
            contour = region.contour
            cv2.drawContours(image_copy, [region.contour], 0, (0, 255, 0), 2)
            if level == Region.LEVEL_CATEGORY:
                label = region.category
            elif level == Region.LEVEL_SUBCATEGORY:
                label = region.subcategory
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
            top_left_corner = contour[:,0].min(), contour[:,1].min()
            right_bottom_corner = contour[:,0].min() + t_size[0] + 3, contour[:,1].min() + t_size[1] + 4

            cv2.rectangle(image_copy, top_left_corner, right_bottom_corner, (0, 255, 0), -1)
            cv2.putText(image_copy, label, (top_left_corner[0], top_left_corner[1] + 32), cv2.FONT_HERSHEY_PLAIN, 3, [225, 255, 255], 2)
        return image_copy
