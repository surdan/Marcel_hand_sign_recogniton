# Marcel_hand_sign_recogniton
基于tensorflow对Marcel静态手势库进行手势识别。

数据集：http://www.idiap.ch/resource/gestures/
本人使用的是网页链接中第三个手势库————————Sebastien Marcel Static Hand Posture Database

20160317更新
2层的卷积神经网络，识别率反而下降到15%

20160327更新
将网络改为densenet,损失函数引入l2正则化，识别率70%左右，最高可达79%。
将数据进行预处理，每个像素值减去全体样本均值。识别率未有改善。
