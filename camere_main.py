import numpy as np
import statsmodels.api as sm
from nnid_zzhowe import nnid_zzhowe


dataset = '/Users/huangzihao/Desktop/work/Image_Quality_Assessment/night_iqa_code_python/dataset'

feat = nnid_zzhowe(dataset)
mos = '/Users/huangzihao/Desktop/work/Image_Quality_Assessment/night_iqa_code_python/mos_30.xlsx'

label = mos
nn,mm = feat.shape
print(nn)
print(mm)