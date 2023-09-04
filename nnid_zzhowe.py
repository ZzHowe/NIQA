import torch
from torchvision import models
import os
import cv2
from nnid_feat import nnid_feat
from torchvision import transforms
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from verify_performance import verify_performance
import matlab.engine
engine = matlab.engine.start_matlab()


def extract_features(image,model):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((227, 227)),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 图像预处理
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 将输入图像传递给模型
    with torch.no_grad():
        model.eval()
        output = model(input_batch)

    # 提取未经flatten操作的特征
    #features = output.view(-1)  # 使用view进行flatten操作

    return output.squeeze().numpy()

def nnid_zzhowe(dataset):

    #pthfile = 'file:///Users/huangzihao/Desktop/work/Image_Quality_Assessment/night_iqa_code_python/squeezenet1_0-a815701f.pth'
    squeezenet = models.squeezenet1_0(pretrained=True, progress=True)
    # state_dict = torch.utils.model_zoo.load_url(pthfile, model_dir=pthfile,
    #                                             map_location=None, progress=True, check_hash=False)
    # squeezenet.load_state_dict(state_dict)  # 读取下载好的模型
    new_model = torch.nn.Sequential(*list(squeezenet.features.children()), squeezenet.classifier)
    #dataset = '/Users/huangzihao/Desktop/work/Image_Quality_Assessment/night_iqa_code_python/dataset'
    print("dataset")
    print(dataset)
    filelist = sorted(os.listdir(dataset))

    error = '.DS_Store'
    if error in filelist:
        filelist.remove('.DS_Store')


    feat = []
    feat_s = []


    for i,filename in enumerate(filelist):
        print(i)
        img_path = dataset + '/' + filename
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        lowlevel_Feat = nnid_feat(img)
        feat.append(lowlevel_Feat)

        high_Features = extract_features(img,new_model)


        feat_s.append(high_Features)

    feat = np.array(feat)
    feat_s = np.array(feat_s)
    return np.concatenate((feat,feat_s),axis=1)

def lyt_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


dataset = '/Users/huangzihao/Desktop/work/Image_Quality_Assessment/night_iqa_code_python/dataset/live30'

feat = nnid_zzhowe(dataset)
#读取excel
excel_path = '/Users/huangzihao/Desktop/work/Image_Quality_Assessment/night_iqa_code_python/dataset/mos_file/mos_30.xlsx'
excel_mos = pd.read_excel(excel_path)
mos = np.array(excel_mos)
print("mos.shape")
print(mos.shape)
print("mos")
print(mos)

label = mos
mm,nn = feat.shape

inst_tmp = np.zeros((mm, nn))

for i in range(nn):
    inst_tmp[:, i] = lyt_normalize(feat[:, i])

feat = inst_tmp
test_train_ratio = 0.8

w = 0.8

total_img = mm
print(total_img)
print(type(total_img))

ssrcc = []
skrcc = []
splcc = []
srmse = []

for k in range(1):
    print(k)
    idx = np.random.permutation(total_img)
    train_idx,test_idx = train_test_split(idx,test_size= 1 - w,random_state=42)
    print(train_idx.shape)
    print(type(train_idx))
    print(test_idx.shape)
    print(type(train_idx))
    print(train_idx.shape[0])
    #train_idx = idx
    train_label = mos[train_idx]
    print("train_label.shape",train_label.shape)
    train_feat = feat[train_idx,:]
    print("train_feat.shape",train_feat.shape)
    test_label = mos[test_idx]
    print("test_label.shape", test_label.shape)
    test_feat = feat[test_idx,:]
    print("test_feat.shape", test_feat.shape)
    svrmodel = SVR()
    svrmodel.fit(train_feat,train_label)
    predict_label = svrmodel.predict(test_feat)
    print("test_idx",train_idx)
    print("predict_label",predict_label)
    ss,kk,pp,rr = verify_performance(test_label,predict_label)
    print(abs(ss))
    print(abs(kk))
    print(abs(pp))
    print(abs(rr))
    ssrcc.append(abs(ss));
    skrcc.append(abs(kk));
    splcc.append(abs(pp));
    srmse.append(abs(rr));


# print("srcc",np.mean(ssrcc))
# print("skrcc",np.mean(skrcc))
# print("splcc",np.mean(splcc))
# print("srmse",np.mean(srmse))
