import matplotlib.pyplot as plt
import cv2
import numpy as np
from pythonRLSA import rlsa


def detect_blocks(img, debug=False, new_width=200):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, depth = img.shape
    img_scale = new_width / width
    new_x, new_y = img.shape[1] * img_scale, img.shape[0] * img_scale
    resized_img = cv2.resize(img, (int(new_x), int(new_y)))
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    (thresh, image_binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # THRESH_BINARY_INV

    image_rlsa_horizontal = rlsa.rlsa(image_binary, True, False, 10)
    image_rlsa_vertical = rlsa.rlsa(image_binary, False, True, 10)

    kernel = np.ones((3, 3), np.uint8)
    bitwise_np = 255 - np.bitwise_and(image_rlsa_vertical, image_rlsa_horizontal)

    dilation_np = cv2.dilate(bitwise_np, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(dilation_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color_img = cv2.cvtColor(dilation_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_img, contours, -1, (0, 255, 0), 3)
    parts = []
    for i, cont in enumerate(contours):
        if hierarchy[0][i][-1] == -1:
            x, y, w, h = cv2.boundingRect(cont)
            # cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            new_x, new_y = int(x / img_scale), int(y / img_scale)
            new_w, new_h = int(w / img_scale), int(h / img_scale)
            part = grey_img[new_y:new_y + new_h, new_x:new_x + new_w]
            parts.append(part)
            cv2.imwrite("data/out_img/part_of_img_{0}.jpg".format(i), part)
            cv2.rectangle(img, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 5)

    cv2.imwrite("data/out_img/part_of_img.jpg", img)
    return parts


def compute_projections(part):
    part_resized = cv2.resize(part, (100, 100))
    x_proj = np.sum(part_resized, axis=0).astype(np.uint8)
    y_proj = np.sum(part_resized, axis=1).astype(np.uint8)
    return x_proj, y_proj
    # plt.plot(x_proj)
    # plt.plot(y_proj)
    # plt.show()


if __name__ == "__main__":
    image = cv2.imread('data/cropped_pdf_img/out_0.jpg')
    parts = detect_blocks(image, 300)
    for part in parts:
        compute_projections(part)
    # cv2.imshow("bitwise img with np after dilation ", return_img)
    # cv2.waitKey(0)

    # cv2.imshow("gwe", return_img)
    # cv2.waitKey(0)
