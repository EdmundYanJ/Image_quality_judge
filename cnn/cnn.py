import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.optim as optim
import math

SIZE=256
count=1500
test_amount=int(count/75)
num_epochs=30
BATCH_SIZE=32

def read_saliency(path):
    coordinate = pd.read_csv(path)
    data_column = list(coordinate.columns)
    form = coordinate[[data_column[0], data_column[1]]]
    form.columns = ['x', 'y',]
    return form

coordinate=read_saliency('../maxPoint.csv')
#print(coordinate.head())

def loaddata(datapath):
    train_count = int(0.8 * count)
    train_data_x = np.empty((train_count,3, 256, 256), dtype="float32")
    train_data_y = []

    test_count = int(0.2 * count)
    test_data_x = np.empty((test_count,3, 256, 256), dtype="float32")
    test_data_y = []

    for i in range(1, int(count/15+1)):
        if i > 0 and i < 10:
            stri = '00' + str(i)
        elif i > 9 and i < 100:
            stri = '0' + str(i)
        elif i == 100:
            stri = str(i)
        path = datapath + '/' + stri+ '/' + stri + '.csv'
        score = pd.read_csv(path)
        data_column = list(score.columns)
        form = score[[data_column[0], data_column[1], data_column[2], data_column[3], data_column[4]]]
        form.columns = ['name', 'color', 'exposure', 'noise', 'texture']
        # print(form.head())

        for j in range(15):
            childpath = datapath + '/'+ stri + '/' + form['name'][j]
            #print(childpath)
            x = coordinate['x'][15 * (i - 1) + j]
            y = coordinate['y'][15 * (i - 1) + j]
            img = cv2.imread(childpath)
            #print(img.shape)
            img=img[x-SIZE//2:x+SIZE//2,y-SIZE//2:y+SIZE//2]
            B,G,R=cv2.split(img)
            imgs=[B,G,R]
            arr = np.asarray(imgs, dtype="float32")
            #print(arr.shape)
            num=15*(i-1)+j
            if num<int(0.8*count):
                train_data_x[num, :, :, :] = arr
                train_data_y.append(float(form['noise'][j]))
            else:
                test_data_x[num-int(0.8*count), :, :, :] = arr
                test_data_y.append(float(form['noise'][j]))
            #print(img.shape)
        print('process %d done'%i)

    train_data_x = train_data_x / 255
    train_data_y = np.asarray(train_data_y)
    train_data_x = torch.from_numpy(train_data_x)
    train_data_y = torch.from_numpy(train_data_y)

    test_data_x = test_data_x / 255
    test_data_y = np.asarray(test_data_y)
    test_data_x = torch.from_numpy(test_data_x)
    test_data_y = torch.from_numpy(test_data_y)

    #print(train_data_x.size(),train_data_y.size(),test_data_x.size(),test_data_y.size())
    return train_data_x, train_data_y,test_data_x,test_data_y

train_data_x, train_data_y,test_data_x,test_data_y=loaddata('../image/training_dataset')
print('Data process done!')

train_dataset=Data.TensorDataset(train_data_x,train_data_y)
test_dataset=Data.TensorDataset(test_data_x,test_data_y)

train_dataloader=Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)


class SimpleNet(nn.Module):
    def __init__(self,):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()#(256,256,12)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()#(256,256,12)

        self.pool1 = nn.MaxPool2d(kernel_size=2)#(128,128,12)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()#(128,128,24)

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()#(128,128,24)

        self.pool2 = nn.MaxPool2d(kernel_size=2)  # (64,64,24)

        self.conv5 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()  # (64,64,48)
        self.conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()  # (64,64,48)

        self.pool3 = nn.MaxPool2d(kernel_size=2)  # (32,32,48)

        self.conv7 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()  # (32,32,96)

        self.conv8 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()  # (32,32,96)

        self.pool4 = nn.MaxPool2d(kernel_size=2)  # (16,16,96)

        self.fc1 = nn.Linear(in_features=16 * 16 * 96, out_features=96)
        self.relu9=nn.ReLU()

        self.fc2 = nn.Linear(in_features= 96, out_features=32 )
        self.relu10 = nn.ReLU()

        self.fc3=nn.Linear(in_features= 32, out_features=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool1(output)

        output = self.conv3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.relu4(output)

        output = self.pool2(output)

        output = self.conv5(output)
        output = self.relu5(output)

        output = self.conv6(output)
        output = self.relu6(output)

        output = self.pool3(output)

        output = self.conv7(output)
        output = self.relu7(output)

        output = self.conv8(output)
        output = self.relu8(output)

        output = self.pool4(output)

        output = output.view(-1, 16 * 16 * 96)

        output = self.fc1(output)
        output = self.relu9(output)

        output = self.fc2(output)
        output = self.relu10(output)


        output = self.fc3(output)

        return output

def srocc(model,test_data,test_label):
    model.eval()
    prediction = model(test_data)
    save1 = []
    save2 = []
    srocc= np.zeros(int(count/75))
    for i in range(test_amount):
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
        sum=0
        for k in range(15):
            sum=sum+(origin[k]-test_result[k])*(origin[k]-test_result[k])
        sub_srocc=abs(1-6*sum/(15*(15*15-1)))
        srocc[i]=sub_srocc
    return srocc.mean(),save1,save2


def train_model(model_name,prediction_name,label_name):
    model=SimpleNet()

    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss=0
        best_acc=0
        for inputs,labels in train_dataloader:
            # 清除所有累积梯度
            optimizer.zero_grad()
            # 用来自测试集的图像预测类
            outputs = model(inputs)
            #数据格式转换
            labels=np.asarray(labels, dtype="float32")
            labels=torch.from_numpy(labels)
            #print(outputs.dtype,labels.dtype)
            # 根据实际标签和预测值计算损失
            loss = loss_fn(outputs, labels)
            # 传播损失
            loss.backward()
            # 根据计算的梯度调整参数
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss=train_loss/(0.8*count)

        acc,save1,save2=srocc(model,test_data_x,test_data_y)
        if acc>best_acc:
            best_acc=acc
            torch.save(model, model_name)
            save1 = pd.DataFrame(data=save1)
            save1.to_csv(prediction_name, encoding='gbk')  # './color/color_prediction.csv'

            save2 = pd.DataFrame(data=save2)
            save2.to_csv(label_name, encoding='gbk')  # './color/color_label.csv'
        #if  (epoch+1)%1==0:
        print('************')
        print('epoch %d'%(epoch+1),': train loss is %f'%train_loss,' acc is %f'%acc)


    print('train model done')
    #torch.save(model, model_name)
    return model

model_name='./result/model/color_CNN12.pkl'
prediction_name='./result/color/color_prediction'+'.csv'
label_name='./result/color/color_label'+'.csv'
model=train_model(model_name,prediction_name,label_name)

def test(model,test_data,test_label,name1,name2):
    model.eval()
    prediction = model(test_data)
    save1 = []
    save2 = []
    for i in range(test_amount):
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

    save1 = pd.DataFrame(data=save1)
    save1.to_csv(name1, encoding='gbk') #'./color/color_prediction.csv'

    save2 = pd.DataFrame(data=save2)
    save2.to_csv(name2, encoding='gbk')#'./color/color_label.csv'
    print('test data done')

prediction_name='./result/color/color_prediction'+'.csv'
label_name='./result/color/color_label'+'.csv'

#test(model,test_data_x,test_data_y,prediction_name,label_name)