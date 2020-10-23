from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from glob import glob
import scipy.misc as sic
import subprocess
import models.network as net
import argparse
import utils.utils as utils
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='Test', type=str, help="Name of model")
parser.add_argument("--save_freq", default=5, type=int, help="save frequency of epochs")
parser.add_argument("--use_gpu", default=1, type=int, help="use gpu or not")
parser.add_argument("--with_IRT", default=0, type=int, help="use IRT or not")
parser.add_argument("--IRT_initialization", default=0, type=int, help="use initialization for IRT or not")
parser.add_argument("--max_epoch", default=25, type=int, help="The max number of epochs for training")
parser.add_argument("--input", default='./demo/colorization/goat_input', type=str, help="dir of input video")
parser.add_argument("--processed", default='./demo/colorization/goat_processed', type=str, help="dir of processed video")
parser.add_argument("--output", default='None', type=str, help="dir of output video")

seed = 2020
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

ARGS = parser.parse_args()
print(ARGS)
save_freq = ARGS.save_freq
input_folder = ARGS.input
processed_folder = ARGS.processed
with_IRT = ARGS.with_IRT
maxepoch = ARGS.max_epoch + 1
model=  ARGS.model
task = "/{}_IRT{}_initial{}".format(model, with_IRT, ARGS.IRT_initialization) #Colorization, HDR, StyleTransfer, Dehazing


def compute_error(real,fake):
    return tf.reduce_mean(tf.abs(fake-real))

def Lp_loss(x, y):
    vgg_real = utils.build_vgg19(x*255.0)
    vgg_fake = utils.build_vgg19(y*255.0,reuse=True) 
    p0=compute_error(vgg_real['input']/255.0,vgg_fake['input']/255.0)
    p1=compute_error(vgg_real['conv1_2']/255.0,vgg_fake['conv1_2']/255.0)/2.6
    p2=compute_error(vgg_real['conv2_2']/255.0,vgg_fake['conv2_2']/255.0)/4.8
    p3=compute_error(vgg_real['conv3_2']/255.0,vgg_fake['conv3_2']/255.0)/3.7
    p4=compute_error(vgg_real['conv4_2']/255.,vgg_fake['conv4_2']/255.)/5.6
    p5=compute_error(vgg_real['conv5_2']/255.,vgg_fake['conv5_2']/255.)*10/1.5
    return p0+p1+p2+p3+p4+p5

def prepare_paired_input(task, id, input_names, processed_names, is_train=0):
    net_in = np.float32(scipy.misc.imread(input_names[id]))/255.0
    if len(net_in.shape) == 2:
        net_in = np.tile(net_in[:,:,np.newaxis], [1,1,3])
    net_gt = np.float32(scipy.misc.imread(processed_names[id]))/255.0
    org_h,org_w = net_in.shape[:2]   
    h = org_h // 32 * 32
    w = org_w // 32 * 32             
    print(net_in.shape, net_gt.shape)
    return net_in[np.newaxis, :h, :w, :], net_gt[np.newaxis, :h, :w, :]


if ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) 
        for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

input_i = tf.placeholder(tf.float32, shape=[None,None,None,3])
input_target = tf.placeholder(tf.float32, shape=[None,None,None,3])
lossDict = {}
objDict={} 
with tf.variable_scope(tf.get_variable_scope()):
    frame_input = input_i[:, :, :, 0:3]
    frame_processed = input_target[:, :, :, 0:3]
    with tf.variable_scope('individual'):
        if not with_IRT:
            frame_out = net.VCN(frame_input, reuse=False)
            lossDict["total"] = Lp_loss(frame_out, frame_processed)
        else:
            net_pred=net.VCN(frame_input, output_channel=6, reuse=False)
            frame_out = net_pred[...,0:3]
            frame_out_minor = net_pred[...,3:6]
            diff_map = tf.reduce_max(tf.abs(frame_out - frame_processed) / (frame_input + 1e-1), axis=3, keep_dims=True)
            diff_map1 = tf.reduce_max(tf.abs(frame_out_minor - frame_processed) / (frame_input + 1e-1), axis=3, keep_dims=True)
            confidence_map = tf.tile(tf.cast(tf.less(diff_map, diff_map1), tf.float32), [1,1,1,3])
            lossDict["total"] = tf.reduce_mean(tf.abs(frame_out - frame_processed)) + 0.9 * tf.reduce_mean(tf.abs(frame_out_minor - frame_processed)) 
            lossDict["confidence_map"] = tf.reduce_mean(tf.abs(frame_out - frame_processed) * confidence_map) + tf.reduce_mean(
                                            tf.abs(frame_out_minor - frame_processed) * (-confidence_map + 1.0))

opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["total"], var_list=[var for var in tf.trainable_variables()])
if with_IRT:
    opt_IRT=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["confidence_map"], var_list=[var for var in tf.trainable_variables()])
print([var for var in tf.trainable_variables() if var.name.startswith('VCRN')])

# If you want to test a sequence of videos, please modify the following two lines.
input_folders = [input_folder]
processed_folders = [processed_folder]

def prepare_data_path():
    input_folders   = sorted(glob("/disk1/chenyang/VideoStablization/data/BTC_test/input/DAVIS/*"))
    processed_folders= [folder.replace('input', 'processed/CycleGAN/photo2ukiyoe/') for folder in input_folders]    
    assert len(input_folders) == len(processed_folders), "The number of input and processed folders are unequal"
    return input_folders, processed_folders

#input_folders, processed_folders = prepare_data_path()

for folder_idx, input_folder in enumerate(input_folders):
    input_names = sorted(glob(input_folders[folder_idx] + "/*"))
    processed_names = sorted(glob(processed_folders[folder_idx] + "/*"))
    if ARGS.output == "None":
        output_folder = "./result/{}".format(task + '/' + input_folder.split("/")[-2] + '/' + input_folder.split("/")[-1]) 
    else:
        output_folder = ARGS.output + "/" + task + '/' + input_folder.split("/")[-1] 
    print(output_folder, input_folders[folder_idx], processed_folders[folder_idx] )
    var_restore = [v for v in tf.trainable_variables()]
    # assert len(input_names) == len(processed_names), "The number of frames is unequal: input-{}  process-{}".format(
    #                                                   len(input_names) == len(processed_names))
    num_of_sample = min(len(input_names), len(processed_names))
    data_in_memory = [None] * num_of_sample                                                 #Speedup
    for id in range(min(len(input_names), len(processed_names))):                           #Speedup
        net_in,net_gt = prepare_paired_input(task, id, input_names, processed_names)        #Speedup
        data_in_memory[id] = [net_in,net_gt]                                                #Speedup

    sess.run([tf.global_variables_initializer()])
    step = 0
    for epoch in range(1,maxepoch):
        print("Processing epoch {}".format(epoch))
        frame_id = 0
        if os.path.isdir("{}/{:04d}".format(output_folder, epoch)):
            continue
        else:
            os.makedirs("{}/{:04d}".format(output_folder, epoch))
        if not os.path.isdir("{}/training".format(output_folder)):
            os.makedirs("{}/training".format(output_folder))

        print(len(input_names), len(processed_names))
        for id in range(num_of_sample): #range(80):#(min(len(input_names), len(processed_names))):
            if with_IRT:      
                if epoch < 6 and ARGS.IRT_initialization:
                    net_in,net_gt = data_in_memory[0]      #Option: 
                    _, out_img, crt_loss = sess.run([opt, frame_out, lossDict["total"]], feed_dict={input_i:net_in, input_target:net_gt})
                else:
                    net_in,net_gt = data_in_memory[id]
                    _, out_img, crt_loss = sess.run([opt_IRT, frame_out, lossDict["confidence_map"]], feed_dict={input_i:net_in, input_target:net_gt})
            else:
                net_in,net_gt = data_in_memory[id] 
                _, out_img, crt_loss = sess.run([opt, frame_out, lossDict["total"]], feed_dict={input_i:net_in, input_target:net_gt})

            frame_id+=1
            step+=1
            if step % 10 == 0:
                print("Image iter: {} {} {} || Loss: {:.4f} ".format(epoch, frame_id, step, crt_loss))
            if step % 100 == 0 :
                sic.imsave("{}/training/step{:06d}_{:06d}.jpg".format(output_folder, step, id), 
                           np.uint8(np.concatenate([net_in[0], out_img[0], net_gt[0]], axis=1).clip(0,1) * 255.0))               



        if epoch % save_freq == 0:
            for id in range(num_of_sample):
                st=time.time()
                net_in,net_gt = data_in_memory[id]
                print("Test: {}-{} \r".format(id, num_of_sample))
                if with_IRT:
                    out_img, out_img1, crt_confidence_map = sess.run([frame_out, frame_out_minor, confidence_map], 
                            feed_dict={input_i:net_in, input_target:net_gt})                
                    sic.imsave("{}/{:04d}/predictions_{:05d}.jpg".format(output_folder, epoch, id),
                        np.uint8(np.concatenate([net_in[0,:,:,:3],out_img[0], out_img1[0],net_gt[0], crt_confidence_map[0]], axis=1).clip(0,1) * 255.0))
                    sic.imsave("{}/{:04d}/out_main_{:05d}.jpg".format(output_folder, epoch, id),np.uint8(out_img[0].clip(0,1) * 255.0))               
                    sic.imsave("{}/{:04d}/out_minor_{:05d}.jpg".format(output_folder, epoch, id),np.uint8(out_img1[0].clip(0,1) * 255.0))               
                else:
                    out_img = sess.run(frame_out, feed_dict={input_i:net_in, input_target:net_gt})                
                    sic.imsave("{}/{:04d}/predictions_{:05d}.jpg".format(output_folder, epoch, id), 
                        np.uint8(np.concatenate([net_in[0,:,:,:3], out_img[0], net_gt[0]],axis=1).clip(0,1) * 255.0))
                    sic.imsave("{}/{:04d}/out_main_{:05d}.jpg".format(output_folder, epoch, id), 
                        np.uint8(out_img[0].clip(0,1) * 255.0))               
