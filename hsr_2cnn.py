# -*- coding=UTF-8 -*-
##卷积池化——卷积池化——全连接
from __future__ import division
import matplotlib
matplotlib.use('AGG')
import os
import random
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



###################################加载数据集#######################################
def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    #获得data_dir的所有子目录，每个代表一个标签
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    #通过标签目录循环，并且收集labels\images两个列表中的数据
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        #对于每个标签，加载其图像并将其添加到图片列表中
        # And add the label number (i.e. directory name) to the labels list.
        #将标签号添加到标签列表中
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = ""
train_data_dir = os.path.join(ROOT_PATH, "/Train")
test_data_dir = os.path.join(ROOT_PATH, "/Test")

images, labels = load_data(train_data_dir)


print("Unique Labels: {0}\nTotal Images: {1}".
      format(len(set(labels)), len(images)))




##############################图片预处理#############################################

######查看原图像大小——————76*66
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

####### 调整图像
images32 = [skimage.transform.resize(image, (32, 32),mode='reflect') for image in images]
##display_images_and_labels(images32, labels)

######查看图像尺寸——————32*32
for image in images32[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

    

############################# 构建图 ############################################# 
labels_a = np.array(labels)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)

# Create a graph to hold the model.
#创建图
graph = tf.Graph()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)      
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W): #卷积使用1步长（stride size）,0边距(padding size)的模板，保证输出是同一大小
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):#池化采用的是最简单的2*2大小的模板做max pooling
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    #输入层
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

 
     #第一层卷积和池化
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(images_ph, [-1,32,32,3]) 
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #第二层卷积和池化
    W_conv2 = weight_variable([5, 5, 32, 64])#卷积在每5*5的patch中得到64个特征，输入通道是32
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#卷积计算
    h_pool2 = max_pool_2x2(h_conv2)

    
    
    #全连接层
    images_flat = tf.contrib.layers.flatten(h_pool2)
    logits = tf.contrib.layers.fully_connected(images_flat , 6, tf.nn.relu)

   
    #输出标签
    predicted_labels = tf.argmax(logits, 1)
 
    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph))
    

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    # TODO: rename to tf.global_variables_initializer() on TF 0.12.
    init = tf.initialize_all_variables()

print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)
############################# 构建图end#############################################
##########################    训练   #############################################

# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
_ = session.run([init])


for i in range(201):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)
    
############################## 预测  ############################################
# Pick 10 random images
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])
plt.show()

############################  评估 ################################################
# Load the test dataset.
test_images, test_labels = load_data(test_data_dir)

# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32),mode='reflect')
                 for image in test_images]
##display_images_and_labels(test_images32, test_labels)

# Run predictions against the full test set.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images32})[0]

# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

accuracy = match_count / len(test_labels)

##correct_prediction = tf.equal(test_labels,predicted)
##accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print("Accuracy: {:.3f}".format(accuracy))

session.close()


   

    
