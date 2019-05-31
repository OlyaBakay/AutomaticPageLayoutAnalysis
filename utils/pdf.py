import os
from pdf2image import convert_from_path

def extract_images(files, out_path, dpi=300):
    for file in files:
        data_path = os.path.join(out_path, os.path.splitext(os.path.basename(file))[0])
        pages = convert_from_path(file, dpi)
        os.makedirs(data_path, exist_ok=True)

        for ind, page in enumerate(pages):
            page.save(os.path.join(data_path, 'img_{}.jpg'.format(ind)), 'JPEG')

if __name__ == "__main__":
    in_path = "../sources/zbirnyk/tom_1/"
    #files = ["../sources/zbirnyk/tom_1/1120-2163-1-PB.pdf"]
    files = [os.path.join(in_path, file) for file in os.listdir(in_path)]
    extract_images(files, "../data/supervisely/zbirnyk/tom_1/")
