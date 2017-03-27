# densenet(1608.06993)
# modify the dense net
# seems before logits we need batch_norm and relu
# and make the weights initializer as MSRA style
# approached an accuracy of 93%
# -*- coding: utf-8 -*-
from sklearn.preprocessing import Normalizer
import re
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils

TOWER_NAME = 'tower'
NUM_CLASSES = 6

slim = tf.contrib.slim
    

def bn_drp_scope(is_training=True, keep_prob=0.8):
  keep_prob = keep_prob if is_training else 1
  print(' bn_drp_scope:is_training:%s, keep_prob:%f' %(
      is_training, keep_prob))
  with slim.arg_scope(
      [slim.batch_norm],
      scale=True, 
      updates_collections=None):
    with slim.arg_scope(
        [slim.dropout],
        is_training=is_training, keep_prob=keep_prob) as bsc:
      return bsc

def densenet_arg_scope(weight_decay=0.004):
  """Defines the default cifarnet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d],
      weights_initializer=
          tf.contrib.layers.variance_scaling_initializer(
              factor=2.0,mode='FAN_IN',uniform=False),
      activation_fn=None, biases_initializer = None, padding = 'same',
      stride=1) as sc:
    return sc



def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
  current = slim.batch_norm(current, scope=scope+'_bn')
  current = tf.nn.relu(current)
  current = slim.conv2d(current, num_outputs, kernel_size, scope=scope+'_conv')
  current = slim.dropout(current, scope=scope+'_dropout')
  return current

def block(net, layers, growth, scope='block'):
  for idx in xrange(layers):
    tmp = bn_act_conv_drp(net, growth, [3,3], 
        scope=scope+'_conv3x3'+str(idx))
    net = tf.concat(axis=3, values=[net, tmp])
  return net

  
def inference(images, is_training=True):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  reduce_dim = lambda x:int(int(x.shape[-1])*compression_rate)
 
  depth = 40
  weight_decay = 1e-4
  layers = 12
  keep_prob = 0.8
  growth = 12
  compression_rate = 1

  end_points = {}
  scope='densenet'
  with tf.variable_scope(scope, 'DenseNet', [images, NUM_CLASSES]):
    with \
        slim.arg_scope(densenet_arg_scope()), \
        slim.arg_scope(bn_drp_scope(is_training=is_training,
            keep_prob=keep_prob)) as ssc:
      scope='conv1'
    
      
      net = slim.conv2d(images, 16,[3,3], scope=scope)
      
      end_points[scope] = net

    

      scope='block1'
      net = block(net, layers, growth, scope=scope)
      end_points[scope] = net
     

      scope='compress1'
      net = bn_act_conv_drp(net, reduce_dim(net), [1,1], scope=scope)
      end_points[scope] = net
      

      scope='avgpool1'
      net = slim.avg_pool2d(net, [2,2], stride=2, scope=scope)
      end_points[scope] = net
      

      scope='block2'
      net = block(net, layers, growth, scope=scope)
      end_points[scope] = net
    

      scope='compress2'
      net = bn_act_conv_drp(net, reduce_dim(net), [1,1], scope=scope)
      end_points[scope] = net
      

      scope='avgpool2'
      net = slim.avg_pool2d(net, [2,2], stride=2, scope=scope)
      end_points[scope] = net
      
      
      scope='block3'
      net = block(net, layers, growth, scope=scope)
      end_points[scope] = net
 

      net = slim.batch_norm(net, scope='before_logits_batchnorm')
      net = tf.nn.relu(net)
      
      
      net = slim.avg_pool2d(net, net.shape[1:3], 
          scope='before_logits_global_avegage')

      biases_initializer = tf.constant_initializer(0.1)
      net = slim.conv2d(net, NUM_CLASSES, [1,1], 
         biases_initializer=biases_initializer)
      softmax_linear = tf.squeeze(net)

  return softmax_linear

