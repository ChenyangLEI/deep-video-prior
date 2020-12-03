from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import scipy.io
import tensorflow as tf
import tf_slim as slim
import numpy as np
from glob import glob
import scipy.misc as sic
import subprocess
import models.network as net
import argparse
import utils.utils as utils
import random
import sys
from art import text2art
import GPUtil
from pathlib import Path
from PIL import Image
from prodict import Prodict

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='Test', type=str, help="Name of model")
parser.add_argument("--save_freq", default=5, type=int, help="save frequency of epochs")
parser.add_argument("--use_gpu", default=1, type=int, help="use gpu or not")
parser.add_argument("--with_IRT", default=0, type=int, help="use IRT or not")
parser.add_argument("--IRT_initialization", default=0, type=int, help="use initialization for IRT or not")
parser.add_argument("--max_epoch", default=25, type=int, help="The max number of epochs for training")
parser.add_argument("--input", default='None', type=str, help="dir of input images")
parser.add_argument("--processed", default='None', type=str, help="dir of processed images")
parser.add_argument("--output", default='None', type=str, help="dir of output images")
parser.add_argument("--format", default='png', type=str, help="format of output image")

seed = 2020
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

ARGS = parser.parse_args()
print("ARGS:{}".format(ARGS))
save_freq = ARGS.save_freq
input_folder = ARGS.input
processed_folder = ARGS.processed
with_IRT = ARGS.with_IRT
maxepoch = ARGS.max_epoch + 1
model=  ARGS.model
task = "{}_IRT{}_initial{}".format(model, with_IRT, ARGS.IRT_initialization) #Colorization, HDR, StyleTransfer, Dehazing

fmt = Prodict(jpg = {'ext':'.jpg', 'format':'JPEG'}, png = {'ext':'.png', 'format':'PNG'})
cur_fmt = fmt[ARGS.format]

print(text2art("TOP LEVEL VARS"))
print("Top level vars: save_freq:{}, input_folder:{}, processed_folder:{}, with_IRT:{}, maxepoch:{}, model:{}, task:{}".format(
    save_freq, input_folder, processed_folder, with_IRT, maxepoch, model, task))

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
    print("\r\tnet_in.shape:{} - net_gt.shape:{}".format(net_in.shape, net_gt.shape))     
    return net_in[np.newaxis, :h, :w, :], net_gt[np.newaxis, :h, :w, :]


if ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) 
        for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

# Below is an alternative way to get access to GPUs, which would allow multiple GPU usage. Method above always chooses the single GPU with the most available memory.
# That said, this code would need a lot of refactoring to leverage multiple GPUs. But it would be good, as it's hard not to run out of memory.

# GPUtil.showUtilization(all=True) #https://github.com/anderskm/gputil
# deviceIDs = GPUtil.getAvailable(order = 'first', limit = 2, maxLoad = 0.5, maxMemory = 0.5, includeNan=True)
# print("deviceIDs:{}".format(deviceIDs))
# os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs)
# print("os.environ[\"CUDA_VISIBLE_DEVICES\"]:{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
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
            lossDict["confidence_map"] = tf.reduce_mean(tf.abs(frame_out - frame_processed) * confidence_map) + tf.reduce_mean(tf.abs(frame_out_minor - frame_processed) * (-confidence_map + 1.0))

opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["total"], var_list=[var for var in tf.trainable_variables()])
if with_IRT:
    opt_IRT=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["confidence_map"], var_list=[var for var in tf.trainable_variables()])

# If you want to test a sequence of videos, please modify the following two lines.
input_folders = [input_folder]
processed_folders = [processed_folder]

for folder_idx, input_folder in enumerate(input_folders):
    input_names = sorted(glob(input_folders[folder_idx] + "/*"))
    processed_names = sorted(glob(processed_folders[folder_idx] + "/*"))

    print("input_names:{}, processed_names:{}".format(input_names, processed_names))
    if ARGS.output == "None":
        output_folder = "result/{}".format(task + '/' + input_folder.split("/")[-2] + '/' + input_folder.split("/")[-1]) 
    else:
        output_folder = ARGS.output + "/" + task + '/' + input_folder.split("/")[-1]

    print(text2art("CHECK PATHS"))
    print("output_folder:{}, input_folders[folder_idx]:{}, processed_folders[folder_idx]:{}".format(output_folder, input_folders[folder_idx], processed_folders[folder_idx]))
    
    var_restore = [v for v in tf.trainable_variables()]
    assert len(input_names) == len(processed_names), "The number of frames is unequal:{}".format(len(input_names) == len(processed_names))
    num_of_sample = min(len(input_names), len(processed_names))
    data_in_memory = [None] * num_of_sample                                                 #Speedup
    for id in range(min(len(input_names), len(processed_names))):                           #Speedup
        net_in,net_gt = prepare_paired_input(task, id, input_names, processed_names)        #Speedup
        data_in_memory[id] = [net_in,net_gt]                                                #Speedup

    sess.run([tf.global_variables_initializer()])
    step = 0
    for epoch in range(1,maxepoch):
        print("Processing epoch: {}, num_of_sample:{}".format(epoch, num_of_sample))
        frame_id = 0
        if os.path.isdir("{}/{:04d}".format(output_folder, epoch)):
            print("ISDIR, continuing")
            continue
        else:
            print("NO DIR, mkdir")
            os.makedirs("{}/{:04d}".format(output_folder, epoch))
        if not os.path.isdir("{}/training".format(output_folder)):
            print("NO TRAINING DIR, mkdir")
            os.makedirs("{}/training".format(output_folder))
            
        for id in range(num_of_sample): #range(80):#(min(len(input_names), len(processed_names))):
            if with_IRT:      
                print("WITH IRT, id:{}".format(id))
                if epoch < 6 and ARGS.IRT_initialization:
                    print("WITH IRT 1, id:{}".format(id))
                    net_in,net_gt = data_in_memory[0]      #Option: 
                    _, out_img, crt_loss = sess.run([opt, frame_out, lossDict["total"]], feed_dict={input_i:net_in, input_target:net_gt})
                else:
                    print("WITH IRT 2, id:{}".format(id))
                    net_in,net_gt = data_in_memory[id]
                    _, out_img, crt_loss = sess.run([opt_IRT, frame_out, lossDict["confidence_map"]], feed_dict={input_i:net_in, input_target:net_gt})
            else:
                print("NO IRT, id:{}".format(id))
                net_in,net_gt = data_in_memory[id] 
                _, out_img, crt_loss = sess.run([opt, frame_out, lossDict["total"]], feed_dict={input_i:net_in, input_target:net_gt})

            frame_id+=1
            step+=1
            if step % 10 == 0:
                print("Epoch:{}, frame_id{}, step:{}, loss: {:.4f}, step % 10: {}, step % 100: {}".format(epoch, frame_id, step, crt_loss, step % 10, step % 100))
            if step % 100 == 0 :
                tr_path = "{}/training/step{:06d}_{:06d}{}".format(output_folder, step, id, cur_fmt.ext)
                tr_img_array = np.uint8(np.concatenate([net_in[0], out_img[0], net_gt[0]], axis=1).clip(0,1) * 255.0)
                print("Saving training image to training_path:{}, uint_8:{}".format(tr_path, tr_img_array))

                # https://stackoverflow.com/a/47292141/869838 - subsampling and quality settings are available if we use Pillow directly, rather than through scipy wrapper          
                # sic.imsave(training_path, training_img_array)     
                Image.fromarray(tr_img_array).save(tr_path, format=cur_fmt.format, subsampling=0, quality=100)


        print("num_of_sample:{}".format(num_of_sample))

        # PIL MODES, for reference - impacts quality of saved image
        # * 'L' (8-bit pixels, black and white)
        # * 'P' (8-bit pixels, mapped to any other mode using a color palette)
        # * 'RGB' (3x8-bit pixels, true color)
        # * 'RGBA' (4x8-bit pixels, true color with transparency mask)
        # * 'CMYK' (4x8-bit pixels, color separation)
        # * 'YCbCr' (3x8-bit pixels, color video format)
        # * 'I' (32-bit signed integer pixels)
        # * 'F' (32-bit floating point pixels)

        if epoch % save_freq == 0:
            for id in range(num_of_sample):
                st=time.time()  
                net_in,net_gt = data_in_memory[id]
                print("Test: {}-{} \r".format(id, num_of_sample))
                if with_IRT:
                    out_img, out_img1, crt_confidence_map = sess.run([frame_out, frame_out_minor, confidence_map], feed_dict={input_i:net_in, input_target:net_gt})                

                    # https://stackoverflow.com/a/47292141/869838 - subsampling and quality settings are available if we use Pillow directly, rather than through scipy wrapper
                    Image.fromarray(np.uint8(np.concatenate([net_in[0,:,:,:3],out_img[0], out_img1[0],net_gt[0], crt_confidence_map[0]], axis=1).clip(0,1) * 255.0)).save(
                        "{}/{:04d}/predictions_{:05d}{}".format(output_folder, epoch, id, cur_fmt.ext), 
                        format=cur_fmt.format, subsampling=0, quality=100)
                    Image.fromarray(np.uint8(out_img[0].clip(0,1) * 255.0)).save(
                        "{}/{:04d}/out_main_{:05d}{}".format(output_folder, epoch, id, cur_fmt.ext), 
                        format=cur_fmt.format, subsampling=0, quality=100)
                    Image.fromarray(np.uint8(out_img1[0].clip(0,1) * 255.0)).save(
                        "{}/{:04d}/out_minor_{:05d}{}".format(output_folder, epoch, id, cur_fmt.ext), 
                        format=cur_fmt.format, subsampling=0, quality=100)

                    print("Save:IRT: id:{} - output_folder:{} \r".format(id, output_folder))     
                else:
                    out_img = sess.run(frame_out, feed_dict={input_i:net_in, input_target:net_gt})                

                    # https://stackoverflow.com/a/47292141/869838 - subsampling and quality settings are available if we use Pillow directly, rather than through scipy wrapper
                    Image.fromarray(np.uint8(np.concatenate([net_in[0,:,:,:3], out_img[0], net_gt[0]],axis=1).clip(0,1) * 255.0)).save(
                        "{}/{:04d}/predictions_{:05d}{}".format(output_folder, epoch, id, cur_fmt.ext), 
                        format=cur_fmt.format, subsampling=0, quality=100)
                    Image.fromarray(np.uint8(out_img[0].clip(0,1) * 255.0)).save(
                        "{}/{:04d}/out_main_{:05d}{}".format(output_folder, epoch, id, cur_fmt.ext), 
                        format=cur_fmt.format, subsampling=0, quality=100)

                    print("Save:NO IRT: id:{} - output_folder:{}".format(id, output_folder))         
                                 
