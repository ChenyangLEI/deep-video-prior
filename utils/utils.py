import numpy as np
import tensorflow as tf
import os
import math
import scipy
import scipy.misc as sic


IMG_EXTENSIONS = [
    '.png', '.PNG', 'jpg', 'JPG', '.jpeg', '.JPEG',
    '.ppm', '.PPM', '.bmp', '.BMP',
]


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

        # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def get_names(dir='./'):
    old_names = os.popen("ls %s"%dir).readlines()
    new_names = [None]*len(old_names)
    for idx in range(len(old_names)):
        new_names[idx] = dir+'/'+old_names[idx][:-1]
    return new_names


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

'''
Return: gray_images, color_images
        gray_images-    [num_frames, H, W, 1]
        color_images-   [num_frames, H, W, 3]
        pixel values [0,1]
'''


def crop_pair_images(net_in, net_gt):
    h_orig,w_orig = net_in.shape[1:3]
    magic = 0.7+0.3*np.random.random()
    h_crop = int(h_orig*magic//32*32)
    w_crop = int(w_orig*magic//32*32)
    w_offset = 0
    h_offset = 0
    try:
        w_offset = np.random.randint(0, w_orig-w_crop-1)
        h_offset = np.random.randint(0, h_orig-h_crop-1)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    net_in=net_in[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    net_gt= net_gt[:,h_offset:h_offset+h_crop,w_offset:w_offset+w_crop,:]
    return net_in, net_gt

def flip_pair_images(im_in, im_GT):
    #Flip
    magic = np.random.random()
    if magic < 0.3:
        im_in=im_in[:,::-1,:,:]
        im_GT = im_GT[:,::-1,:,:]
    magic = np.random.random()
    if magic < 0.3:
        im_in=im_in[:,:,::-1,:]
        im_GT = im_GT[:,:,::-1,:]
    return im_in, im_GT

def resize_pair_images(net_in, net_gt):
    magic = np.random.random()
    if magic < 0.3:
        net_in = net_in[:,::2,::2,:]
        net_gt = net_gt[:,::2,::2,:]     
    return net_in, net_gt


def flip_images(X):
    num_img = X.shape[0]
    magic=np.random.random()
    if magic < 0.3:
        for i in range(num_img):
            X[i,...] = np.fliplr(X[i,...])
    magic=np.random.random()
    if magic < 0.3:
        for i in range(num_img):
            X[i,...] = np.fliplr(X[i,...])
    return X

def pad_images(X,a):
    num_img = X.shape[0]
    h_orig,w_orig = X.shape[1:3]
    newX = np.ones((num_img,a,a,3))
    for i in range(num_img):
        pad_width=((0,a-h_orig),(0,a-w_orig),(0,0))
        newX[i,...] = np.pad(X[i,...],pad_width,'constant')
    return newX

def crop_images(X,a,b,is_sq=False):
    h_orig,w_orig = X.shape[1:3]
    w_crop = np.random.randint(a, b)
    r = w_crop/w_orig
    h_crop = np.int(h_orig*r)
    try:
        w_offset = np.random.randint(0, w_orig-w_crop-1)
        h_offset = np.random.randint(0, h_orig-h_crop-1)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    return X[:,h_offset:h_offset+h_crop-1,w_offset:w_offset+w_crop-1,:]

def degamma(X):
    return np.power(X, 2.2)

def gamma(X):
    return np.power(X, 1/2.2)

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

vgg_rawnet=scipy.io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')
def build_vgg19(input,reuse=False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_rawnet['layers'][0]
        net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
        net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
        return net

def build(input):
    vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
    for layer_id in range(1,6):#6
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    return input

def build_nlayer(input, nlayer):
    vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
    for layer_id in range(1,nlayer):#6
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    return input


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def conv_2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
