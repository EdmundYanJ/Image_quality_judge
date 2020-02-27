# Image_quality_judge
## Dependencies
python 3.6

Pytorch 0.4.1

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
 - `pandas`<br>
 - `numpy`<br>
 - `cv2`<br>
## Data
从[QA4Camera](https://qa4camera.github.io/)下载数据集，并保存在image文件夹下。数据集为1500张分别由15个手机拍摄的图片，共100个场景，每个手机在一个场景中拍一张，由专家从曝光/颜色/纹理/噪声四个方面排名。目标是使用算法自动对图片排名，最终得出不同手机在不同场景下拍摄的优劣。
## Training&Test
### cnn
 - `cd cnn<br>`
 - `python cnn.py`
### traditional_method
 - `cd traditional_method<br>`
 - `python xxx.py`
## Score
score文件夹下保存了由model1/2/3所得出的图像质量评价相关分数各项指标均在0～1之间，越靠近1表示模型越好。<br>
其中以srocc为主要评价指标，因为预测的图片排名与实际图片排名秩相关性越高则srocc越高。
## Alogrithm
### cnn
由于图片数据集小，图片分辨率高（4032 * 3024），所以先使用显著性模型分割图片，取出图像中人最容易关注的区域（256 * 256）<br>
然后放入卷积神经网络中提取特征，最后得到图片的分数，再对图片进行排名。
### traditional_method
Cross_validate是使用5折交叉验证法对算法进行评估。<br>
model1/2/3分别是使用GLCM(灰度共生矩阵)/hsv色彩空间下的方差/rgb和yuv空间下的方差和图像熵，使用全连接神经网络进行训练所得的模型。
