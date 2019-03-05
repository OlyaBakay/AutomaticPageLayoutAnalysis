import textract

# res = textract.process('data/Zapysky_Tom_001.pdf').decode('utf-8')

res = textract.process('data/khronika/Khronika_Ch_01.pdf').decode('utf-8')
with open('data/res.txt', 'w') as the_file:
    the_file.write(res)
# the_file.close()


# def extract_txt(file_path):
#     text = textract.process(file_path, method='tesseract').decode('utf-8')
#     outfn = file_path[:-4] + '.txt'
#     with open(outfn, 'wb') as output_file:
#         output_file.write(text)
#     return file_path
#
#
# extract_txt('data/Zapysky_Tom_001.pdf')
#
