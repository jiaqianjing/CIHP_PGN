from __future__ import print_function
from datetime import datetime
import os
import scipy.io as sio
import cv2
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
from utils import *
import pathlib


def read_image_list(data_dir):
    img_path = pathlib.Path(data_dir)
    img_list = [str(i) for i in list(img_path.glob('*.jpg'))]
    return img_list

N_CLASSES = 20
DATA_DIR = '../../gans/VITON-HD/datasets/test/image/'
RESTORE_FROM = './checkpoint/CIHP_pgn'
NUM_STEPS = len(read_image_list(DATA_DIR))
IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)


def read_images_from_disk():
    image_list = read_image_list(DATA_DIR)
    print(f"image_list: {image_list}")
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    queue = tf.train.slice_input_producer([images], shuffle=False) 
    img_contents = tf.read_file(queue[0])
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    return img

def main():
    """Create the model and start the evaluation process."""
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    with tf.name_scope("create_inputs"):
        image = read_images_from_disk()
        
        # image shape ------> (?, ?, 3)
        print('image shape ------>', image.shape)
        # 沿着 axis=[1] 翻转，即左右（W）翻转
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = read_image_list(DATA_DIR)

    image_batch = tf.stack([image, image_rev])
    
    # image_batch --> (2, ?, ?, 3)
    print('image_batch -->', image_batch.shape)
    
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch050 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.50)), tf.to_int32(tf.multiply(w_orig, 0.50))]))
    image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
    image_batch125 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.25)), tf.to_int32(tf.multiply(w_orig, 1.25))]))
    image_batch150 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.50)), tf.to_int32(tf.multiply(w_orig, 1.50))]))
    image_batch175 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.75)), tf.to_int32(tf.multiply(w_orig, 1.75))]))

    # image_batch050: shape (2, ?, ?, 3)
    print('image_batch050: shape', image_batch050.shape) 
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)
    
    # parsing net
    parsing_out1_050 = net_050.layers['parsing_fc']
    parsing_out1_075 = net_075.layers['parsing_fc']
    parsing_out1_100 = net_100.layers['parsing_fc']
    parsing_out1_125 = net_125.layers['parsing_fc']
    parsing_out1_150 = net_150.layers['parsing_fc']
    parsing_out1_175 = net_175.layers['parsing_fc']

    parsing_out2_050 = net_050.layers['parsing_rf_fc']
    parsing_out2_075 = net_075.layers['parsing_rf_fc']
    parsing_out2_100 = net_100.layers['parsing_rf_fc']
    parsing_out2_125 = net_125.layers['parsing_rf_fc']
    parsing_out2_150 = net_150.layers['parsing_rf_fc']
    parsing_out2_175 = net_175.layers['parsing_rf_fc']

    # edge net
    edge_out2_100 = net_100.layers['edge_rf_fc']
    edge_out2_125 = net_125.layers['edge_rf_fc']
    edge_out2_150 = net_150.layers['edge_rf_fc']
    edge_out2_175 = net_175.layers['edge_rf_fc']

    # parsing_out1_050 ----> (2, ?, ?, 20)  parsing_out2_050 ----> (2, ?, ?, 20)
    print("parsing_out1_050 ---->", parsing_out1_050.shape)
    print("parsing_out2_050 ---->", parsing_out2_050.shape)

    # edge_out2_100 ----> (2, ?, ?, 1)
    print("edge_out2_100 ---->", edge_out2_100.shape)

    # combine resize
    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize_images(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)

    # parsing_out1 ----> (2, ?, ?, 20)  parsing_out2 ----> (2, ?, ?, 20)
    print("parsing_out1 ---->", parsing_out1.shape)
    print("parsing_out2 ---->", parsing_out2.shape)

    edge_out2_100 = tf.image.resize_images(edge_out2_100, tf.shape(image_batch)[1:3,])
    edge_out2_125 = tf.image.resize_images(edge_out2_125, tf.shape(image_batch)[1:3,])
    edge_out2_150 = tf.image.resize_images(edge_out2_150, tf.shape(image_batch)[1:3,])
    edge_out2_175 = tf.image.resize_images(edge_out2_175, tf.shape(image_batch)[1:3,])
    edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)

    # edge_out2_100 ----> (2, ?, ?, 1)
    print("edge_out2_100 ---->", edge_out2_100.shape)
    # edge_out2 ----> (2, ?, ?, 1)
    print("edge_out2 ---->", edge_out2.shape)
                                           
    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    # raw_output ----> (2, ?, ?, 20)
    print("raw_output ---->", raw_output.shape)
    # head_output ----> (?, ?, 20)
    print("head_output ---->", head_output.shape)
    # tail_output ----> (?, ?, 20)
    print("tail_output ---->", tail_output.shape)

    # return list: [(?, ?, 1), ..., (?, ?, 1)]  对应像素点的类别 score
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    # raw_output_all ----> (1, ?, ?, 20)
    print("raw_output_all ---->", raw_output_all.shape)
    
    # 每个像素点对应类别的最大 score: pred_scores ----> (1, ?, ?)
    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    print("pred_scores ---->", pred_scores.shape)
    # 每个像素点对应类别的下标: raw_output_all ----> (1, ?, ?)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    print("raw_output_all ---->", raw_output_all.shape)
    # pred_all ----> (1, ?, ?, 1)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.
    print("pred_all ---->", pred_all.shape)


    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    # raw_edge ----> (2, ?, ?, 1)
    print("raw_edge ---->", raw_edge.shape)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
    # raw_edge_all ----> (1, ?, ?, 1)
    print("raw_edge_all ---->", raw_edge_all.shape)
    pred_edge = tf.sigmoid(raw_edge_all)
    res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)
    # res_edge ----> (1, ?, ?, 1)
    print("res_edge ---->", res_edge.shape)

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # evaluate prosessing
    parsing_dir = '../../gans/VITON-HD/datasets/test/image-parse'
    if not os.path.exists(parsing_dir):
        os.makedirs(parsing_dir)
    # Iterate over training steps.
    for step in range(NUM_STEPS):
        parsing_, scores, edge_, = sess.run([pred_all, pred_scores, pred_edge])
        print('step {:d}'.format(step))
        print (image_list[step])
        img_split = image_list[step].split('/')
        img_id = img_split[-1][:-4]

        # 输出带有像素点对应类别的标注的灰度图
        cv2.imwrite('{}/{}.png'.format(parsing_dir, img_id), parsing_[0,:,:,0])

    coord.request_stop()
    coord.join(threads)
    


if __name__ == '__main__':
    main()


##############################################################333
