# -*- coding=UTF-8 -*-
"""将图片scale变为（-1,1）,RGB图像"""

from __future__ import division
from sklearn.preprocessing import Normalizer
import os
import random
import skimage.data
import skimage.transform
import numpy as np
import tensorflow as tf
import dense as model
#model = __import__('origin')
slim = tf.contrib.slim

# 运行图形嵌入到notebook中
##%matplotlib inline
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('train', True,
                           """training mode.""")
tf.app.flags.DEFINE_string('checkpoint_path', 'ckpt/', '')




#######################加载数据集#################################
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

"""将图片scale归一化到（-1,1）"""
def scalechange(x):
    x = (x-0.5)*2
    return x
"""图片每个样本减去全体均值"""
def vggscalechange(x):
    x = x-mean
    return x

# Load training and testing datasets.
ROOT_PATH = "/opt/Project-1/"

train_data_dir = os.path.join(ROOT_PATH, "Train")
test_data_dir = os.path.join(ROOT_PATH, "Test")
images, labels = load_data(train_data_dir)


print("Unique Labels: {0}\nTotal Images: {1}".
      format(len(set(labels)), len(images)))


######查看原图像大小——————、
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

####### 调整图像
images32 = [skimage.transform.resize(image, (32, 32),mode='reflect') for image in images]

mean = np.mean(images32)

images32_n = []
for image in images32:
    images32_n.append(vggscalechange(image))


test_images, test_labels = load_data(test_data_dir)

# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32),mode='reflect')
                 for image in test_images]
test_images32_n = []
for image in test_images32:
    test_images32_n.append(vggscalechange(image))

######查看图像尺寸——————
for image in images32_n[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
##############################图片预处理end#############################################
shuffled_indices = range(len(images32_n))


#############################最小可行模型############################################# 
labels_a = np.array(labels)
images_a = np.array(images32_n)

labels_t = np.array(test_labels)
images_t = np.array(test_images32)


print("labels: %s\nimages: %s" % 
    (labels_a.shape, images_a.shape))

def get_batch(size=128):
  random.shuffle(shuffled_indices)
  sample_indexes = shuffled_indices[:size]
  sample_images = [images_a[i] for i in sample_indexes]
  sample_labels = [labels_a[i] for i in sample_indexes]
  return sample_images, sample_labels


# Create a graph to hold the model.
#创建图
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
   
    #输入层
    
    images_ph = tf.placeholder(tf.float32, [None, 32, 32,3])
    labels_ph = tf.placeholder(tf.int32, [None])
    labels_one_hot = tf.one_hot(labels_ph, 6, on_value=1, off_value=0, dtype=tf.int32)

    logits = model.inference(images_ph, 
        is_training=FLAGS.train)

    predicted_labels = tf.argmax(logits, 1)
    labels_int64 = tf.cast(labels_ph, tf.int64)
    
    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels_one_hot))

    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    alpha =tf.divide(tf.pow(l2_loss,2), 
                tf.multiply(loss, 250))
    alpha = 1e-3
    total_loss = loss + alpha*l2_loss

    accuracy_op = tf.contrib.metrics.accuracy(predicted_labels, labels_int64)
    

    # Create training op.
    loss_opt = tf.train.AdamOptimizer(
        learning_rate=0.001).minimize(loss)
    regular_opt = tf.train.AdamOptimizer(
        learning_rate=0.001).minimize(total_loss)
    train = tf.group(loss_opt, regular_opt)
     
    
    total_loss = tf.Print(total_loss, 
        [total_loss, loss, l2_loss])
    saver = tf.train.Saver(tf.global_variables(), 
        max_to_keep=5)
    init_op = tf.global_variables_initializer()
    # And, finally, an initialization op to execute before training.
    # TODO: rename to tf.global_variables_initializer() on TF 0.12.

#############################最小可行模型end#############################################
##########################    训练   #############################################

# Create a session to run the graph we created
with  tf.Session(graph=graph) as sess:
  print('init')
  # First step is always to initialize all variables. 
  # We don't care about the return value, though. It's None.
  if FLAGS.train:
      
    sess.run(init_op)
    if not os.path.exists(FLAGS.checkpoint_path):
      os.makedirs(FLAGS.checkpoint_path)
    print('training starting...')
    for i in range(3801):
      batch_images, batch_labels = get_batch()
      _, loss_value, accuracy_value= \
        sess.run([train, total_loss, accuracy_op],
            feed_dict={images_ph: batch_images, 
                       labels_ph: batch_labels})
      if i % 10 == 0:
        print("epoch %d, Loss: %f, accuracy: %f" %
            ( i,  loss_value, accuracy_value))
        eval_accuracy = sess.run(accuracy_op,
            feed_dict={images_ph:images_t, 
                       labels_ph: test_labels})
        print("eval accuracy: %f" % eval_accuracy)
      if i % 100 == 0:
        saver.save(sess, os.path.join(FLAGS.checkpoint_path,
            model.__name__), 
            global_step = i)
  
    print('training ended.')
    ##################    训练 end  #############      
  else:
    #################     预测  ###############
    # restore checkpoints

    ckpt = tf.train.get_checkpoint_state(
        FLAGS.checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      print(ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
    
    print('evaluating test set...')
    # Load the test dataset.
    test_images, test_labels = load_data(test_data_dir)

    try:
      accuracy_value = sess.run(accuracy_op, 
        feed_dict={images_ph:images_t,
                   labels_ph:test_labels})
    except Exception as ex:
      print(ex)
    print("accuracy: %f"%accuracy_value)


print('done.')
#########################  评估 end ####################################################
