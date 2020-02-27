import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

FEATURES = 5

#获取图片的熵
def entropy(path):
    img = cv2.imread(path)  # (4032, 3024)
    image = cv2.imread(path, 0)
    hist = cv2.calcHist([image],
                        [0],  # 使用的通道
                        None,  # 没有使用mask
                        [256],  # HistSize
                        [0.0, 255.0])  # 直方图柱的范围
    hist = hist / (image.size)
    H = 0
    for i in range(256):
        if hist[i] <= 0:
            pass
        else:
            H -= hist[i] * math.log(hist[i], 2)
    # print(mean,stddv)
    return H

#获取像素的均值和方差
def getstddv(path):
    img=cv2.imread(path)  # (4032, 3024)
    #img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    mean1 , stddv1 = cv2.meanStdDev(img)
    img_yuv=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    mean2,stddv2=cv2.meanStdDev(img_yuv[0:256,0:256])
    noise=(mean2[0]+0.001)/(stddv2[0]+0.001)
    #print(mean,stddv)
    return mean1 , stddv1  ,noise

# path = '/Users/yanjunbing/Desktop/python_workspace/image_judge/image/testing_set/001/A_001.jpg'
# mean,stddv=getstddv(path)

# 将数据集全部转换成特征向量形式
data = []
color = []
exposure = []
noise = []
texture = []


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
            mean, stddv,n = getstddv(childpath)
            for k in range(3):
                #subdata[2 * k] = (float(mean[k]))
                subdata[k] = (float(stddv[k]))
            subdata[3] = float(entropy(childpath))
            subdata[4] = float(n)
            #print(subdata)
            data.append(subdata)
            color.append(form['color'][j])
            exposure.append(form['exposure'][j])
            noise.append(form['noise'][j])
            texture.append(form['texture'][j])
            # print(data,form['color'][j])
        print('process', i, 'done')
    print('data process done!')


make_traindata()

train_data = data[0:1200]
test_data = data[1200:1500]
color_train_label = color[0:1200]
color_test_label = color[1200:1500]
exposure_train_label = exposure[0:1200]
exposure_test_label = exposure[1200:1500]
noise_train_label = noise[0:1200]
noise_test_label = noise[1200:1500]
texture_train_label = texture[0:1200]
texture_test_label = texture[1200:1500]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(5, 10)
        self.hidden2 = torch.nn.Linear(10, 20)
        self.hidden3 = torch.nn.Linear(20, 40)
        self.predict = torch.nn.Linear(40, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        #x=F.relu(x)
        x = self.predict(x)
        return x


# color
color_net = Net()
print(color_net)
optimizer = torch.optim.Adam(color_net.parameters())
loss_func = torch.nn.MSELoss()

color_train_label = np.array(color_train_label)
color_train_label = torch.from_numpy(color_train_label).float()
train_data = torch.stack(train_data).to()
for epoch in range(2000):
    prediction = color_net(train_data)
    loss = loss_func(prediction, color_train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print_loss = loss.item()
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        # print('acc is {:.4f}'.format(acc))  # 精度
torch.save(color_net, './model/color.pkl')

test_data = torch.stack(test_data).to()
prediction = color_net(test_data)

save1 = []
save2 = []
for i in range(20):
    my_result = []
    test_result = []
    for j in range(15):
        my_result.append(float(prediction[i * 15 + j]))
        test_result.append(color_test_label[i * 15 + j])
    origin = my_result
    my_result = np.array(my_result)
    my_result = np.argsort(my_result)
    for j in range(15):
        origin[my_result[j]] = j + 1
    origin = np.array(origin)
    test_result = np.array(test_result)
    save1.append(origin)
    save2.append(test_result)

save1 = pd.DataFrame(data=save1)
save1.to_csv('./color/color_prediction.csv', encoding='gbk')

save2 = pd.DataFrame(data=save2)
save2.to_csv('./color/color_label.csv', encoding='gbk')

print('color done')

# exposure
exposure_net = Net()
print(exposure_net)
optimizer = torch.optim.Adam(exposure_net.parameters())
loss_func = torch.nn.MSELoss()

exposure_train_label = np.array(exposure_train_label)
exposure_train_label = torch.from_numpy(exposure_train_label).float()
# train_data=torch.stack(train_data).to()
for epoch in range(2000):
    prediction = exposure_net(train_data)
    loss = loss_func(prediction, exposure_train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print_loss = loss.item()
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        # print('acc is {:.4f}'.format(acc))  # 精度
torch.save(exposure_net, './model/exposure.pkl')

# test_data=torch.stack(test_data).to()
prediction = exposure_net(test_data)

save1 = []
save2 = []
for i in range(20):
    my_result = []
    test_result = []
    for j in range(15):
        my_result.append(float(prediction[i * 15 + j]))
        test_result.append(exposure_test_label[i * 15 + j])
    origin = my_result
    my_result = np.array(my_result)
    my_result = np.argsort(my_result)
    for j in range(15):
        origin[my_result[j]] = j + 1
    origin = np.array(origin)
    test_result = np.array(test_result)
    save1.append(origin)
    save2.append(test_result)

save1 = pd.DataFrame(data=save1)
save1.to_csv('./exposure/exposure_prediction.csv', encoding='gbk')

save2 = pd.DataFrame(data=save2)
save2.to_csv('./exposure/exposure_label.csv', encoding='gbk')

print('exposure done')

# noise
noise_net = Net()
print(noise_net)
optimizer = torch.optim.Adam(noise_net.parameters())
loss_func = torch.nn.MSELoss()

noise_train_label = np.array(noise_train_label)
noise_train_label = torch.from_numpy(noise_train_label).float()
# train_data=torch.stack(train_data).to()
for epoch in range(2000):
    prediction = noise_net(train_data)
    loss = loss_func(prediction, noise_train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print_loss = loss.item()
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        # print('acc is {:.4f}'.format(acc))  # 精度
torch.save(noise_net, './model/noise.pkl')

# test_data=torch.stack(test_data).to()
prediction = noise_net(test_data)

save1 = []
save2 = []
for i in range(20):
    my_result = []
    test_result = []
    for j in range(15):
        my_result.append(float(prediction[i * 15 + j]))
        test_result.append(noise_test_label[i * 15 + j])
    origin = my_result
    my_result = np.array(my_result)
    my_result = np.argsort(my_result)
    for j in range(15):
        origin[my_result[j]] = j + 1
    origin = np.array(origin)
    test_result = np.array(test_result)
    save1.append(origin)
    save2.append(test_result)

save1 = pd.DataFrame(data=save1)
save1.to_csv('./noise/noise_prediction.csv', encoding='gbk')

save2 = pd.DataFrame(data=save2)
save2.to_csv('./noise/noise_label.csv', encoding='gbk')

print('noise done')

# texture
texture_net = Net()
print(texture_net)
optimizer = torch.optim.Adam(texture_net.parameters())
loss_func = torch.nn.MSELoss()

texture_train_label = np.array(texture_train_label)
texture_train_label = torch.from_numpy(texture_train_label).float()
# train_data=torch.stack(train_data).to()
for epoch in range(2000):
    prediction = texture_net(train_data)
    loss = loss_func(prediction, texture_train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print_loss = loss.item()
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        # print('acc is {:.4f}'.format(acc))  # 精度
torch.save(texture_net, './model/texture.pkl')

# test_data=torch.stack(test_data).to()
prediction = texture_net(test_data)

save1 = []
save2 = []
for i in range(20):
    my_result = []
    test_result = []
    for j in range(15):
        my_result.append(float(prediction[i * 15 + j]))
        test_result.append(texture_test_label[i * 15 + j])
    origin = my_result
    my_result = np.array(my_result)
    my_result = np.argsort(my_result)
    for j in range(15):
        origin[my_result[j]] = j + 1
    origin = np.array(origin)
    test_result = np.array(test_result)
    save1.append(origin)
    save2.append(test_result)

save1 = pd.DataFrame(data=save1)
save1.to_csv('./texture/texture_prediction.csv', encoding='gbk')

save2 = pd.DataFrame(data=save2)
save2.to_csv('./texture/texture_label.csv', encoding='gbk')

print('texture done')
