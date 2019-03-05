from pdf2image import convert_from_path
import numpy as np
import os


def extract_images(in_path, out_path):
    pages = convert_from_path(in_path, 500)

    os.makedirs(out_path, exist_ok=True)

    for ind, page in enumerate(pages):
        print(np.array(ind))
        page.save(os.path.join(out_path, 'out_{}.jpg'.format(ind)), 'JPEG')


if __name__ == "__main__":
    extract_images("data/zapysky/Zapysky_Tom_001-pages-6.pdf", "data/cropped_pdf_img")
