import os
from PIL import Image

dir = '/Users/huangzihao/Desktop/work/UWIQA'
new_dir = '/Users/huangzihao/Desktop/work/Image_Quality_Assessment/night_iqa_code_python/dataset/UWIQA'

filelist = os.listdir(dir)

for i,filename in enumerate(filelist):
    path = dir + '/' + filename
    img = Image.open(path)
    idx = "{:03d}".format(i)
    new_path = new_dir + '/' + idx + '.png'
    img.save(new_path)

print("finish")
