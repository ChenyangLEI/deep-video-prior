from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import subprocess
from tensorflow.contrib.layers import layer_norm



def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def lrelu(x):
    return tf.maximum(x*0.2,x)

def bilinear_up_and_concat(x1, x2, output_channels, in_channels, scope):
    with tf.variable_scope(scope):
        upconv = tf.image.resize_images(x1, [tf.shape(x1)[1]*2, tf.shape(x1)[2]*2] )
        upconv.set_shape([None, None, None, in_channels])
        # upconv = slim.conv2d(upconv,output_channels,[3,3], rate=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),scope='up_conv1')# Why?
        upconv = slim.conv2d(upconv,output_channels,[3,3], rate=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=scope)
        upconv_output =  tf.concat([upconv, x2], axis=3)
        upconv_output.set_shape([None, None, None, output_channels*2])
    return upconv_output


def bilinear_up(x1, x2, output_channels, in_channels, scope):
    with tf.variable_scope(scope):
        upconv = tf.image.resize_images(x1, [tf.shape(x1)[1]*2, tf.shape(x1)[2]*2] )
        upconv.set_shape([None, None, None, in_channels])
        # upconv = slim.conv2d(upconv,output_channels,[3,3], rate=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),scope='up_conv1')# Why?
        upconv = slim.conv2d(upconv,output_channels,[3,3], rate=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=scope)
        upconv_output =  upconv
        #upconv_output.set_shape([None, None, None, output_channels*2])
    return upconv_output

def Encoder(input, channel=32, output_channel=3,reuse=False,ext="",div_num=1):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv5_2')
    up6 =  bilinear_up( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv6_2')
    up7 =  bilinear_up( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv7_2')
    up8 =  bilinear_up( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv8_2')
    up9 =  bilinear_up( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9,output_channel*div_num,[3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_2')
    return conv9

def VCN(input, channel=32, output_channel=3,reuse=False,ext="",div_num=1):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv5_2')
    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9,output_channel*div_num,[3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_2')
    return conv9

def VCRN(input, channel=32, output_channel=3,reuse=False,ext="VCRN"):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv5_2')
    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"r_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"r_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"r_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"r_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv9_1')
    conv9=slim.conv2d(conv9,output_channel,[3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv9_2')
    return conv9

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)
    
def FCN_Dial(input,channel=64, output_channel=3):
    net=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,output_channel,[1,1],rate=1,activation_fn=None,scope='g_conv_last') 
    return net






def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv_filter = tf.cast(deconv_filter,x1.dtype)
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def ResUnet(input, dims=32, output_channel=3, demosaic=False):
    conv1=slim.conv2d(input,dims,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    conv1=slim.conv2d(conv1,dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm, scope='g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv2_1')
    conv2=slim.conv2d(conv2,dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv3_1')
    conv3=slim.conv2d(conv3,dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv4_1')
    conv4=slim.conv2d(conv4,dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,dims*16,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv5_1')
    conv5=slim.conv2d(conv5,dims*16,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv5_2')

    net = conv5
    for i in range(16):
        temp = net
        net = slim.conv2d(net, dims*16, [3,3], activation_fn=lrelu,normalizer_fn=layer_norm, scope='g_res%d_conv1'%i)
        net = slim.conv2d(net, dims*16, [3,3], activation_fn=None,normalizer_fn=layer_norm, scope='g_res%d_conv2'%i)
        net = net + temp  
      
    net = slim.conv2d(net, dims*16, [3,3], activation_fn=None,normalizer_fn=layer_norm, scope='g_res')
    conv5 = net + conv5

    up6 =  upsample_and_concat( conv5, conv4, dims*8, dims*16  )
    conv6=slim.conv2d(up6,  dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,dims*8,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv6_2')

    up7 =  upsample_and_concat( conv6, conv3, dims*4, dims*8  )
    conv7=slim.conv2d(up7,  dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,dims*4,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv7_2')

    up8 =  upsample_and_concat( conv7, conv2, dims*2, dims*4 )
    conv8=slim.conv2d(up8,  dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,dims*2,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv8_2')

    up9 =  upsample_and_concat( conv8, conv1, dims, dims*2 )
    conv9=slim.conv2d(up9,  dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,dims,[3,3], rate=1, activation_fn=lrelu,normalizer_fn=layer_norm,scope='g_conv9_2')
    
    if demosaic:
        conv10 = slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
        conv10 = tf.depth_to_space(conv10,2) 
    else:
        conv10 = slim.conv2d(conv9,output_channel,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    
    return conv10
