import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
SIZE=256
FEATURES = 10
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
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


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

    asm, con, eng, idm = feature_computer(glcm_0)

    return [asm, con, eng, idm]

#获取像素值的均值和方差
def getstddv(path,x,y):
    img=cv2.imread(path)  # (4032, 3024)
    #print(img.shape,x,y)
    #print(path,x-SIZE//2,x+SIZE//2,y-SIZE//2,y+SIZE//2)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean , stddv = cv2.meanStdDev(img_hsv[x-SIZE//2+1:x+SIZE//2-1,y-SIZE//2+1:y+SIZE//2-1])
    #print(mean,stddv)
    return mean,stddv
# path = '/Users/yanjunbing/Desktop/python_workspace/image_judge/image/testing_set/001/A_001.jpg'
# mean,stddv=getstddv(path)

#获取由显著性模型所得到的图像区域（256*256）
def read_saliency(path):
    coordinate = pd.read_csv(path)
    data_column = list(coordinate.columns)
    form = coordinate[[data_column[0], data_column[1]]]
    form.columns = ['x', 'y',]
    return form

coordinate=read_saliency('../maxPoint.csv')

# 将数据集全部转换成特征向量形式
data = []
color = []
exposure = []
noise = []
texture = []

#处理图像数据和标签
def make_traindata():
    fatherpath = '../image/training_dataset/'
    for i in range(1, 101):
        if i > 0 and i < 10:
            stri = '00' + str(i)
        elif i > 9 and i < 100:
            stri = '0' + str(i)
        elif i == 100:
            stri = str(i)
        path = fatherpath + stri + '/' + stri + '.csv'
        score = pd.read_csv(path)
        data_column = list(score.columns)
        form = score[[data_column[0], data_column[1], data_column[2], data_column[3], data_column[4]]]
        form.columns = ['name', 'color', 'exposure', 'noise', 'texture']
        # print(form.head())
        for j in range(15):
            subdata = torch.zeros(FEATURES)
            childpath = fatherpath + stri + '/' + form['name'][j]
            x = coordinate['x'][15 * (i - 1) + j]
            y = coordinate['y'][15 * (i - 1) + j]
            result = GLCM(childpath, x, y)
            mean, stddv = getstddv(childpath, x, y)
            # print(result)
            for k in range(4):
                subdata[k] = result[k]
            for k in range(4,7):
                subdata[k] = float(mean[k-4])
            for k in range(7,10):
                subdata[k] = float(stddv[k-7])
            data.append(subdata)
            color.append(form['color'][j])
            exposure.append(form['exposure'][j])
            noise.append(form['noise'][j])
            texture.append(form['texture'][j])
            # print(data,form['color'][j])
        print('process', i, 'done')
    print('data process done!')

make_traindata()

#创建交叉验证集
def cross_validate(data,color,exposure,noise,texture):
    subset=[[],[],[],[],[]]
    train_data=[[],[],[],[],[]]
    test_data=[[],[],[],[],[]]
    color_subset=[[],[],[],[],[]]
    color_train_label = [[],[],[],[],[]]
    color_test_label = [[],[],[],[],[]]
    exposure_subset=[[],[],[],[],[]]
    exposure_train_label = [[],[],[],[],[]]
    exposure_test_label = [[],[],[],[],[]]
    noise_subset=[[],[],[],[],[]]
    noise_train_label = [[],[],[],[],[]]
    noise_test_label = [[],[],[],[],[]]
    texture_subset=[[],[],[],[],[]]
    texture_train_label = [[],[],[],[],[]]
    texture_test_label = [[],[],[],[],[]]
    for i in range(5):
        subset[i]=data[300*i:300*(i+1)]
        color_subset[i]=color[300*i:300*(i+1)]
        exposure_subset[i] =exposure[300*i:300*(i+1)]
        noise_subset[i] =noise[300*i:300*(i+1)]
        texture_subset[i] =texture[300*i:300*(i+1)]
    for i in range(5):
        test_data[i]=subset[i]
        color_test_label[i]=color_subset[i]
        exposure_test_label[i]=exposure_subset[i]
        noise_test_label[i]=noise_subset[i]
        texture_test_label[i]=texture_subset[i]
        for j in range(5):
            if i!=j:
                train_data[i]+=subset[j]
                color_train_label[i] += color_subset[i]
                exposure_train_label[i] += exposure_subset[i]
                noise_train_label[i] += noise_subset[i]
                texture_train_label[i] += texture_subset[i]
    return train_data,test_data,color_train_label,color_test_label,exposure_train_label,exposure_test_label,noise_train_label,noise_test_label,texture_train_label,texture_test_label

train_data,test_data,color_train_label,color_test_label,exposure_train_label,exposure_test_label,noise_train_label,noise_test_label,texture_train_label,texture_test_label=cross_validate(data,color,exposure,noise,texture)

for i in range(5):
    train_data[i] = torch.stack(train_data[i]).to()
    test_data[i] = torch.stack(test_data[i]).to()
    color_train_label[i] = torch.from_numpy(np.array(color_train_label[i])).float()
    exposure_train_label[i] = torch.from_numpy(np.array(exposure_train_label[i])).float()
    noise_train_label[i] = torch.from_numpy(np.array(noise_train_label[i])).float()
    texture_train_label[i] = torch.from_numpy(np.array(texture_train_label[i])).float()

#创建网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(10, 10)
        # self.hidden2 = torch.nn.Linear(20, 24)
        # self.hidden3 = torch.nn.Linear(24, 6)
        self.predict = torch.nn.Linear(10, 1)

    def forward(self,x):
            x=self.hidden1(x)
            x = F.relu(x)
            #x = self.hidden2(x)
            #x = F.tanh(x)
            #x = self.hidden3(x)
            #x=F.relu(x)
            x=self.predict(x)
            return x

#训练模型
def train_model(net,train_data, train_label, name):
    for epoch in range(2000):
        prediction = net(train_data)
        loss = loss_func(prediction, train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print_loss = loss.item()
            print('*' * 10)
            print('epoch {}'.format(epoch + 1))  # 训练轮数
            print('loss is {:.4f}'.format(print_loss))  # 误差
            # print('acc is {:.4f}'.format(acc))  # 精度
    torch.save(net, name) #'./model/color.pkl'

#测试模型
def test_model(net,test_data,test_label,name1,name2):
    prediction = net(test_data)
    save1 = []
    save2 = []
    #将分数转换为排名
    for i in range(20):
        my_result = []
        test_result = []
        for j in range(15):
            my_result.append(float(prediction[i * 15 + j]))
            test_result.append(test_label[i * 15 + j])
        origin = my_result
        my_result = np.array(my_result)
        my_result = np.argsort(my_result)
        for j in range(15):
            origin[my_result[j]] = j + 1
        origin = np.array(origin)
        test_result = np.array(test_result)
        save1.append(origin)
        save2.append(test_result)

    #保存数据
    save1 = pd.DataFrame(data=save1)
    save1.to_csv(name1, encoding='gbk') #'./color/color_prediction.csv'

    save2 = pd.DataFrame(data=save2)
    save2.to_csv(name2, encoding='gbk')#'./color/color_label.csv'

# color
for i in range(5):
    color_net = Net()
    print(color_net)
    optimizer = torch.optim.Adam(color_net.parameters())
    loss_func = torch.nn.MSELoss()
    model_name='./result/model/color'+str(i+1)+'.pkl'
    train_model(color_net,train_data[i],color_train_label[i],model_name)
    prediction_name='./result/color/color_prediction'+str(i+1)+'.csv'
    label_name='./result/color/color_label'+str(i+1)+'.csv'
    test_model(color_net,test_data[i],color_test_label[i],prediction_name,label_name)
print('color done')

# exposure
for i in range(5):
    exposure_net = Net()
    print(exposure_net)
    optimizer = torch.optim.Adam(exposure_net.parameters())
    loss_func = torch.nn.MSELoss()
    model_name = './result/model/exposure' + str(i + 1) + '.pkl'
    train_model(exposure_net, train_data[i], exposure_train_label[i], model_name)
    prediction_name = './result/exposure/exposure_prediction' + str(i + 1) + '.csv'
    label_name = './result/exposure/exposure_label' + str(i + 1) + '.csv'
    test_model(exposure_net, test_data[i], exposure_test_label[i], prediction_name, label_name)
print('exposure done')

# noise
for i in range(5):
    noise_net = Net()
    print(noise_net)
    optimizer = torch.optim.Adam(noise_net.parameters())
    loss_func = torch.nn.MSELoss()
    model_name = './result/model/noise' + str(i + 1) + '.pkl'
    train_model(noise_net, train_data[i], noise_train_label[i], model_name)
    prediction_name = './result/noise/noise_prediction' + str(i + 1) + '.csv'
    label_name = './result/noise/noise_label' + str(i + 1) + '.csv'
    test_model(noise_net, test_data[i], noise_test_label[i], prediction_name, label_name)
print('noise done')

# texture
for i in range(5):
    texture_net = Net()
    print(texture_net)
    optimizer = torch.optim.Adam(texture_net.parameters())
    loss_func = torch.nn.MSELoss()
    model_name = './result/model/texture' + str(i + 1) + '.pkl'
    train_model(texture_net, train_data[i], texture_train_label[i], model_name)
    prediction_name = './result/texture/texture_prediction' + str(i + 1) + '.csv'
    label_name = './result/texture/texture_label' + str(i + 1) + '.csv'
    test_model(texture_net, test_data[i], texture_test_label[i], prediction_name, label_name)
print('texture done')


