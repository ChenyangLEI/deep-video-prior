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
import subprocess
import models.network as net
import argparse
import utils.utils as utils
import random
from PIL import Image
import json
from prodict import Prodict

# Save log
# One folder, one result

parser = argparse.ArgumentParser()
parser.add_argument("--task", default='Test', type=str, help="Name of task")
parser.add_argument("--input", default='./demo/consistency/Enhancement/bike-packing-input', type=str, help="Dir of input video")
parser.add_argument("--processed", default='./demo/consistency/Enhancement/bike-packing-processed', type=str, help="Dir of processed video")
parser.add_argument("--output", default='Output', type=str, help="Dir of output video")
parser.add_argument("--use_gpu", default=1, type=int, help="Use gpu or not")

# Training options
parser.add_argument('--loss', type=str, default='perceptual',
                            help='Chooses which loss to use. perceptual, l1, l2',
                            choices=["perceptual", "l1", "l2"])
parser.add_argument('--network', type=str, default='unet',
                            help='Chooses which model to use. unet, fcn',
                            choices=["unet"])
parser.add_argument("--coarse_to_fine_speedup", default=1, type=int, help="Use coarse_to_fine_speedup for training")
parser.add_argument("--with_IRT", default=0, type=int, help="Sse IRT or not, set this to 1 if you want to solve multimodal inconsistency")
parser.add_argument("--IRT_initialization", default=1, type=int, help="Sse initialization for IRT")
parser.add_argument("--large_video", default=0, type=int, help="Set this to 1 when the number of video frames are large, e.g., more than 1000 frames")


# logging/saving options
parser.add_argument("--save_freq", default=25, type=int, help="Save frequency of epochs")
parser.add_argument("--max_epoch", default=25, type=int, help="The max number of epochs for training")
parser.add_argument("--format", default='png', type=str, help="Format of output image")



seed = 2020
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

ARGS = parser.parse_args()
print(ARGS)
input_folder = ARGS.input
processed_folder = ARGS.processed
with_IRT = ARGS.with_IRT
maxepoch = ARGS.max_epoch + 1
coarse_to_fine_speedup=ARGS.coarse_to_fine_speedup
task = "/{}".format(ARGS.task) #Colorization, HDR, StyleTransfer, Dehazing

fmt = Prodict(jpg = {'ext':'.jpg', 'format':'JPEG'}, png = {'ext':'.png', 'format':'PNG'})
cur_fmt = fmt[ARGS.format]


os.makedirs("./result/{}".format(ARGS.task), exist_ok=True)
with open("./result/{}/commandline_args.txt".format(ARGS.task) , 'w') as f:
    json.dump(ARGS.__dict__, f, indent=2)


def compute_error(real,fake):
    return tf.reduce_mean(tf.abs(fake-real))

def perceptual_loss(x, y):
    vgg_real = utils.build_vgg19(x * 255.0)
    vgg_fake = utils.build_vgg19(y * 255.0,reuse=True) 
    p0=compute_error(vgg_real['input']/255.0,vgg_fake['input']/255.0)
    p1=compute_error(vgg_real['conv1_2']/255.0,vgg_fake['conv1_2']/255.0)/2.6
    p2=compute_error(vgg_real['conv2_2']/255.0,vgg_fake['conv2_2']/255.0)/4.8
    p3=compute_error(vgg_real['conv3_2']/255.0,vgg_fake['conv3_2']/255.0)/3.7
    p4=compute_error(vgg_real['conv4_2']/255.,vgg_fake['conv4_2']/255.)/5.6
    p5=compute_error(vgg_real['conv5_2']/255.,vgg_fake['conv5_2']/255.)*10/1.5
    return p0+p1+p2+p3+p4+p5

def prepare_paired_input(task, id, input_names, processed_names, is_train=0):
    net_in = np.float32(np.array(Image.open(input_names[id]))) / 255.0
    if len(net_in.shape) == 2:
        net_in = np.tile(net_in[:,:,np.newaxis], [1,1,3])
    net_gt = np.float32(np.array(Image.open(processed_names[id]))) / 255.0
    org_h,org_w = net_in.shape[:2]   
    h = org_h // 32 * 32
    w = org_w // 32 * 32             
    # print(net_in.shape, net_gt.shape)
    return net_in[np.newaxis, :h, :w, :], net_gt[np.newaxis, :h, :w, :]

# def get_loss(output, processed, output_minor=None):
def get_model(input):
    if ARGS.network == "unet":
        return net.VCN(input, reuse=False)
if ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) 
        for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

with tf.variable_scope(ARGS.input+'input'):
    input_i = tf.placeholder(tf.float32, shape=[None,None,None,3])
    input_target = tf.placeholder(tf.float32, shape=[None,None,None,3])
lossDict = {}
objDict={} 
with tf.variable_scope(tf.get_variable_scope()):
    frame_input = input_i[:, :, :, 0:3]
    frame_processed = input_target[:, :, :, 0:3]
    with tf.variable_scope(ARGS.input+'individual'):
        if not with_IRT:
            frame_out = get_model(frame_input)
        else:
            net_pred=net.VCN(frame_input, output_channel=6, reuse=False)
            frame_out = net_pred[...,0:3]
            frame_out_minor = net_pred[...,3:6]
            tf.summary.image("minor mode", tf.clip_by_value(frame_out_minor, 0, 1))

        tf.summary.image("main mode", tf.clip_by_value(frame_out, 0, 1))        
    with tf.variable_scope(ARGS.input+'loss'):
        if not with_IRT:
            # lossDict["total"] = perceptual_loss(frame_out, frame_processed)
            # lossDict["total"] = tf.reduce_mean(tf.abs(frame_out - frame_processed))
            if ARGS.loss == "l2": 
                lossDict["total"] = tf.reduce_mean(tf.square(frame_out - frame_processed))
            elif ARGS.loss == "l1": 
                lossDict["total"] = tf.reduce_mean(tf.abs(frame_out - frame_processed))
            else:
                lossDict["total"] = perceptual_loss(frame_out, frame_processed)
        else:
            diff_map = tf.reduce_max(tf.abs(frame_out - frame_processed) / (frame_input + 1e-1), axis=3, keep_dims=True)
            diff_map1 = tf.reduce_max(tf.abs(frame_out_minor - frame_processed) / (frame_input + 1e-1), axis=3, keep_dims=True)
            confidence_map = tf.tile(tf.cast(tf.less(diff_map, diff_map1), tf.float32), [1,1,1,3])
            lossDict["total"] = tf.reduce_mean(tf.abs(frame_out - frame_processed)) + 0.9 * tf.reduce_mean(tf.abs(frame_out_minor - frame_processed)) 
            lossDict["confidence_map"] = tf.reduce_mean(tf.abs(frame_out - frame_processed) * confidence_map) + tf.reduce_mean(
                                            tf.abs(frame_out_minor - frame_processed) * (-confidence_map + 1.0))
        tf.summary.scalar('loss', lossDict["total"])


opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["total"], var_list=[var for var in tf.trainable_variables()])
if with_IRT:
    opt_IRT=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["confidence_map"], var_list=[var for var in tf.trainable_variables()])
print([var for var in tf.trainable_variables() if var.name.startswith('VCRN')])

merged = tf.summary.merge_all()
writer=tf.summary.FileWriter('./logs/{}'.format(task + '/' + ARGS.output), sess.graph)
sess.run([tf.global_variables_initializer()])
# If you want to test a sequence of videos, please modify the following two lines.


input_names = sorted(glob(ARGS.input + "/*"))
processed_names = sorted(glob(ARGS.processed + "/*"))
output_folder = "./result/{}/{}".format(ARGS.task, ARGS.output) 
print(output_folder, ARGS.input, ARGS.processed)
var_restore = [v for v in tf.trainable_variables()]
# assert len(input_names) == len(processed_names), "The number of frames is unequal: input-{}  process-{}".format(
#                                                   len(input_names) == len(processed_names))

num_of_sample = min(len(input_names), len(processed_names))
data_in_memory = [[None, None]] * num_of_sample      
                                           #Speedup
if not ARGS.large_video:
    for id in range(min(len(input_names), len(processed_names))):                           #Speedup
        net_in,net_gt = prepare_paired_input(task, id, input_names, processed_names)        #Speedup
        data_in_memory[id] = [net_in,net_gt]                                                #Speedup

step = 0
for epoch in range(1,maxepoch):
    print("Processing epoch {}".format(epoch))
    frame_id = 0
    if os.path.isdir("{}/{:04d}".format(output_folder, epoch)):
        continue
    else:
        os.makedirs("{}/{:04d}".format(output_folder, epoch), exist_ok=True)
        os.makedirs("{}/{:04d}_vis".format(output_folder, epoch), exist_ok=True)
    os.makedirs("{}/training".format(output_folder), exist_ok=True)

    print(len(input_names), len(processed_names))
    
    for id in range(num_of_sample): #range(80):#(min(len(input_names), len(processed_names))):
        st = time.time()
        if with_IRT and epoch < 6 and ARGS.IRT_initialization:
            id = 0 
        net_in, net_gt = data_in_memory[id] if not ARGS.large_video else prepare_paired_input(task, id, input_names, processed_names)
        if coarse_to_fine_speedup and epoch < (maxepoch * 4 // 5) : # coarse to fine speed up
            net_in, net_gt = net_in[:,::2,::2,:], net_gt[:,::2,::2,:]
        st = time.time()
        _, crt_loss = sess.run([opt, lossDict["total"]], feed_dict={input_i:net_in, input_target:net_gt})        
        
        frame_id+=1
        step+=1
        if step % 10 == 0:
            print("Image iter: {} {} {} || Loss: {:.4f} {:.6f}".format(epoch, frame_id, step, crt_loss, time.time() - st))
            tf.contrib.summary.scalar('loss', lossDict['total'])
        if step % 50 == 0 :
            summary, out_img, crt_loss = sess.run([merged, frame_out, lossDict["total"]], feed_dict={input_i:net_in, input_target:net_gt})
            writer.add_summary(summary, epoch * num_of_sample + frame_id)
            # writer.add_summary(crt_loss, epoch * num_of_sample + id)
            # save_image = np.uint8(np.concatenate([net_in[0], out_img[0], net_gt[0]], axis=1).clip(0,1) * 255.0)
            # Image.fromarray(save_image).save("{}/training/step{:06d}_{:06d}.jpg".format(output_folder, step, id))


    if epoch % ARGS.save_freq == 0:
        for id in range(num_of_sample):
            st=time.time()
            net_in,net_gt = data_in_memory[id] if not ARGS.large_video else prepare_paired_input(task, id, input_names, processed_names)
            print("Test: {}/{} \r".format(id, num_of_sample))
            if with_IRT:
                out_img, out_img1, crt_confidence_map = sess.run([frame_out, frame_out_minor, confidence_map], 
                        feed_dict={input_i:net_in, input_target:net_gt})        

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

            else:
                out_img = sess.run(frame_out, feed_dict={input_i:net_in, input_target:net_gt})                
                # save_image = np.uint8(out_img[0].clip(0,1) * 255.0)
                # Image.fromarray(save_image).save("{}/{:04d}/out_main_{:05d}.jpg".format(output_folder, epoch, id))        
                # save_image = np.uint8(np.concatenate([net_in[0,:,:,:3], out_img[0], net_gt[0]], axis=1).clip(0,1) * 255.0)
                # Image.fromarray(save_image).save("{}/{:04d}_vis/predictions_{:05d}.jpg".format(output_folder, epoch, id))        

                # https://stackoverflow.com/a/47292141/869838 - subsampling and quality settings are available if we use Pillow directly, rather than through scipy wrapper
                Image.fromarray(np.uint8(np.concatenate([net_in[0,:,:,:3], out_img[0], net_gt[0]],axis=1).clip(0,1) * 255.0)).save(
                    "{}/{:04d}/predictions_{:05d}{}".format(output_folder, epoch, id, cur_fmt.ext), 
                    format=cur_fmt.format, subsampling=0, quality=100)
                Image.fromarray(np.uint8(out_img[0].clip(0,1) * 255.0)).save(
                    "{}/{:04d}/out_main_{:05d}{}".format(output_folder, epoch, id, cur_fmt.ext), 
                    format=cur_fmt.format, subsampling=0, quality=100)

                print("Save:NO IRT: id:{} - output_folder:{}".format(id, output_folder))         

writer.close()