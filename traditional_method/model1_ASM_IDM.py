import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
FEATURES=2
SIZE=256
gray_level = 16

def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    #print(height,width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            #Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            # if p[i][j] > 0.0:
            #     Eng += p[i][j] * math.log(p[i][j])
    return Asm, Idm


def GLCM(image_name,x,y):
    img = cv2.imread(image_name)
    try:
        img_shape = img.shape
    except:
        print('imread error')
        return
    img=img[x-SIZE//2+1:x+SIZE//2-1,y-SIZE//2+1:y+SIZE//2-1]


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0 = getGlcm(img_gray, 1, 0)
    # glcm_1=getGlcm(src_gray, 0,1)
    # glcm_2=getGlcm(src_gray, 1,1)
    # glcm_3=getGlcm(src_gray, -1,1)

    asm,  idm = feature_computer(glcm_0)

    return [asm, idm]

#获得显著性区域
#path = '/Users/yanjunbing/Desktop/python_workspace/image_judge/image/testing_set/001/A_001.jpg'
#mean,stddv=getstddv(path)
def read_saliency(path):
    coordinate = pd.read_csv(path)
    data_column = list(coordinate.columns)
    form = coordinate[[data_column[0], data_column[1]]]
    form.columns = ['x', 'y',]
    return form

coordinate=read_saliency('../maxPoint.csv')
#print(coordinate['x'].size)
#将数据集全部转换成特征向量形式
data=[]
color=[]
exposure=[]
noise=[]
texture=[]
#数据处理，读取图片及标签
def make_traindata():
    fatherpath='../image/training_dataset/'
    for i in range(1,101):
        if i>0 and i<10:
            stri='00'+str(i)
        elif i>9 and i<100:
            stri = '0' + str(i)
        elif i==100:
            stri = str(i)
        path=fatherpath+stri+'/'+stri+'.csv'
        score = pd.read_csv(path)
        data_column = list(score.columns)
        form = score[[data_column[0], data_column[1], data_column[2], data_column[3], data_column[4]]]
        form.columns = ['name', 'color', 'exposure', 'noise', 'texture']
        #print(form.head())
        for j in range(15):
            subdata=torch.zeros(FEATURES)
            childpath=fatherpath+stri+'/'+form['name'][j]
            x=coordinate['x'][15*(i-1)+j]
            y=coordinate['y'][15*(i-1)+j]
            #print(15*(i-1)+j)
            result=GLCM(childpath,x,y)
            #print(result)
            for k in range(2):
                subdata[k]=result[k]
            data.append(subdata)
            color.append(form['color'][j])
            exposure.append(form['exposure'][j])
            noise.append(form['noise'][j])
            texture.append(form['texture'][j])
            #print(data,form['color'][j])
        print('process',i,'done')
    print('data process done!')

make_traindata()

train_data=data[0:1200]
test_data=data[1200:1500]
color_train_label=color[0:1200]
color_test_label=color[1200:1500]
exposure_train_label=exposure[0:1200]
exposure_test_label=exposure[1200:1500]
noise_train_label=noise[0:1200]
noise_test_label=noise[1200:1500]
texture_train_label=texture[0:1200]
texture_test_label=texture[1200:1500]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden1=torch.nn.Linear(2,8)
        self.hidden2 = torch.nn.Linear(8, 4)
        #self.hidden3 = torch.nn.Linear(16, 4)
        self.predict=torch.nn.Linear(4,1)

    def forward(self,x):
        x=self.hidden1(x)
        x = self.hidden2(x)
        #x = self.hidden3(x)
        #x=F.relu(x)
        x=self.predict(x)
        return x

#color
color_net=Net()
print(color_net)
optimizer=torch.optim.Adam(color_net.parameters())
loss_func=torch.nn.MSELoss()

color_train_label=np.array(color_train_label)
color_train_label=torch.from_numpy(color_train_label).float()
train_data=torch.stack(train_data).to()
for epoch in range(2000):
    prediction=color_net(train_data)
    loss=loss_func(prediction,color_train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print_loss = loss.item()
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        #print('acc is {:.4f}'.format(acc))  # 精度
torch.save(color_net, './model/color.pkl')

test_data=torch.stack(test_data).to()
prediction=color_net(test_data)

save1=[]
save2=[]
for i in range(20):
    my_result=[]
    test_result =[]
    for j in range(15):
        my_result.append(float(prediction[i*15+j]))
        test_result.append(color_test_label[i*15+j])
    origin=my_result
    my_result=np.array(my_result)
    my_result=np.argsort(my_result)
    for j in range(15):
        origin[my_result[j]]=j+1
    origin=np.array(origin)
    test_result=np.array(test_result)
    save1.append(origin)
    save2.append(test_result)

save1 = pd.DataFrame(data=save1)
save1.to_csv('./color/color_prediction.csv', encoding='gbk')

save2 = pd.DataFrame(data=save2)
save2.to_csv('./color/color_label.csv', encoding='gbk')

print('color done')

#exposure
exposure_net=Net()
print(exposure_net)
optimizer=torch.optim.Adam(exposure_net.parameters())
loss_func=torch.nn.MSELoss()

exposure_train_label=np.array(exposure_train_label)
exposure_train_label=torch.from_numpy(exposure_train_label).float()
#train_data=torch.stack(train_data).to()
for epoch in range(2000):
    prediction=exposure_net(train_data)
    loss=loss_func(prediction,exposure_train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print_loss = loss.item()
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        #print('acc is {:.4f}'.format(acc))  # 精度
torch.save(exposure_net, './model/exposure.pkl')

#test_data=torch.stack(test_data).to()
prediction=exposure_net(test_data)

save1=[]
save2=[]
for i in range(20):
    my_result=[]
    test_result =[]
    for j in range(15):
        my_result.append(float(prediction[i*15+j]))
        test_result.append(exposure_test_label[i*15+j])
    origin=my_result
    my_result=np.array(my_result)
    my_result=np.argsort(my_result)
    for j in range(15):
        origin[my_result[j]]=j+1
    origin=np.array(origin)
    test_result=np.array(test_result)
    save1.append(origin)
    save2.append(test_result)

save1 = pd.DataFrame(data=save1)
save1.to_csv('./exposure/exposure_prediction.csv', encoding='gbk')

save2 = pd.DataFrame(data=save2)
save2.to_csv('./exposure/exposure_label.csv', encoding='gbk')

print('exposure done')

#noise
noise_net=Net()
print(noise_net)
optimizer=torch.optim.Adam(noise_net.parameters())
loss_func=torch.nn.MSELoss()

noise_train_label=np.array(noise_train_label)
noise_train_label=torch.from_numpy(noise_train_label).float()
#train_data=torch.stack(train_data).to()
for epoch in range(2000):
    prediction=noise_net(train_data)
    loss=loss_func(prediction,noise_train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print_loss = loss.item()
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        #print('acc is {:.4f}'.format(acc))  # 精度
torch.save(noise_net, './model/noise.pkl')

#test_data=torch.stack(test_data).to()
prediction=noise_net(test_data)

save1=[]
save2=[]
for i in range(20):
    my_result=[]
    test_result =[]
    for j in range(15):
        my_result.append(float(prediction[i*15+j]))
        test_result.append(noise_test_label[i*15+j])
    origin=my_result
    my_result=np.array(my_result)
    my_result=np.argsort(my_result)
    for j in range(15):
        origin[my_result[j]]=j+1
    origin=np.array(origin)
    test_result=np.array(test_result)
    save1.append(origin)
    save2.append(test_result)

save1 = pd.DataFrame(data=save1)
save1.to_csv('./noise/noise_prediction.csv', encoding='gbk')

save2 = pd.DataFrame(data=save2)
save2.to_csv('./noise/noise_label.csv', encoding='gbk')

print('noise done')

#texture
texture_net=Net()
print(texture_net)
optimizer=torch.optim.Adam(texture_net.parameters())
loss_func=torch.nn.MSELoss()

texture_train_label=np.array(texture_train_label)
texture_train_label=torch.from_numpy(texture_train_label).float()
#train_data=torch.stack(train_data).to()
for epoch in range(2000):
    prediction=texture_net(train_data)
    loss=loss_func(prediction,texture_train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print_loss = loss.item()
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        #print('acc is {:.4f}'.format(acc))  # 精度
torch.save(texture_net, './model/texture.pkl')

#test_data=torch.stack(test_data).to()
prediction=texture_net(test_data)

save1=[]
save2=[]
for i in range(20):
    my_result=[]
    test_result =[]
    for j in range(15):
        my_result.append(float(prediction[i*15+j]))
        test_result.append(texture_test_label[i*15+j])
    origin=my_result
    my_result=np.array(my_result)
    my_result=np.argsort(my_result)
    for j in range(15):
        origin[my_result[j]]=j+1
    origin=np.array(origin)
    test_result=np.array(test_result)
    save1.append(origin)
    save2.append(test_result)

save1 = pd.DataFrame(data=save1)
save1.to_csv('./texture/texture_prediction.csv', encoding='gbk')

save2 = pd.DataFrame(data=save2)
save2.to_csv('./texture/texture_label.csv', encoding='gbk')

print('texture done')