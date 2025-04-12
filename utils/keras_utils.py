from __future__ import absolute_import, division, print_function

"""
this file contains helper code for setting keras parameters
"""

import os

import numpy as np
import tensorflow as tf


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s == None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum([np.prod(p.shape) for p in model.trainable_weights])
    non_trainable_count = sum([np.prod(p.shape) for p in model.non_trainable_weights])

    # Total memory (float32 = 4 bytes)
    total_memory = (
        4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    )
    gbytes = np.round(total_memory / (1024.0**3), 3)  # Convert to GB
    return gbytes


def set_keras_parms(threads=0, gpu_mem=2 * 1024, gpu_fraction=0.3):
    """Assume that you have 2GB of GPU memory"""

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if gpu_fraction < 0.1:
        gpu_fraction = 0.1

    gpus = tf.config.list_physical_devices("GPU")
    print("$" * 20, gpus)

    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=int(gpu_fraction * gpu_mem)
                    )
                ],
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    if threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)


def set_keras_growth(gpunumbers: str = "0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpunumbers
    if "," in gpunumbers:
        gpunumbers = map(int, gpunumbers.split(","))
    else:
        gpunumbers = [int(gpunumbers)]

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpunumber in gpunumbers:
                tf.config.experimental.set_memory_growth(gpus[gpunumber], True)
                print(f"GPU {gpunumbers} memory growth enabled")
        except RuntimeError as e:
            print(f"Error enabling GPU memory growth: {e}")

    tf.random.set_seed(1)


def manhattan_distance(A, B):
    return tf.keras.ops.sum(tf.keras.ops.abs(A - B), axis=1, keepdims=True)


# if network output is softmax, the total output of the net is 2 x sum(softmax), i.e. 1, see page 4 of chopra2005learning
upper_b = 2


def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    print(x.shape[1])


# input = tf.placeholder(tf.float32)


# from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    sqaure_pred = tf.keras.ops.square(y_pred)
    margin_square = tf.keras.ops.square(tf.keras.ops.maximum(margin - y_pred, 0))
    return tf.keras.ops.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


# y_pred is a vector of vector of concated outputs from G(x), these vectors
# needs to be unconcated and calculate the L1 norm between each of them
# Then this needs to be combined using eq 9 of chopra2005
def chopraloss(y_pred, y_true):
    # g_shape = tf.shape(y_pred)[1]
    # g_length = tf.constant(tf.round(tf.divide(g_shape,tf.constant(2))))
    tf.Print(y_pred, [tf.shape(y_pred)], "shape = ")
    g_one, g_two = tf.split(y_pred, 2, axis=0)
    # g_one = y_pred[:,0:g_length].eval()
    # g_two = y_pred[:,g_length+1:g_length*2].eval()
    # E_w = || G_w(X_1) - G_w(X_2) ||
    ml = tf.norm(tf.subtract(g_one, g_two), ord=1)
    # ml = manhattan_distance(y_pred,y_true)
    y = y_true
    l_g = 1
    l_l = 1
    q = upper_b
    thisshape = tf.shape(y_pred)  # .shape
    ones = tf.ones(thisshape, tf.float32)
    negy = ones - y
    return (y * (2.0 / q) * (ml**2)) + (negy * 2 * q * tf.exp(-(2.77 / q) * ml))


# y_pred is a vector of vector of concated outputs from G(x), these vectors
# needs to be unconcated and calculate the L1 norm between each of them
# Then this needs to be combined using eq 9 of chopra2005
def chopraloss3(y_pred, y_true):
    # g_shape = tf.shape(y_pred)[1]
    # g_length = tf.constant(tf.round(tf.divide(g_shape,tf.constant(2))))
    tf.Print(y_pred, [tf.shape(y_pred)], "shape = ")
    g_one, g_two = tf.split(y_pred, 2, axis=0)
    # g_one = y_pred[:,0:g_length].eval()
    # g_two = y_pred[:,g_length+1:g_length*2].eval()
    # E_w = || G_w(X_1) - G_w(X_2) ||
    ml = tf.keras.ops.abs(g_one - g_two)
    # ml = manhattan_distance(y_pred,y_true)
    y = tf.keras.ops.round(ml)
    l_g = 1
    l_l = 1
    q = upper_b
    thisshape = y_pred.shape
    ones = tf.keras.ops.ones_like(g_one)
    # negy = (ones-y)
    part_one = (ones - y) * (2.0 / q) * tf.keras.ops.square(ml)
    part_two = y * 2 * q * tf.keras.ops.exp(-(2.77 / q) * ml)
    return part_one + part_two


def chopraloss2(y_pred, y_true):
    l_g = 1
    l_l = 1
    q = upper_b
    total = 0
    for i in zip(y_pred, y_true):
        i_ypred = i[0]
        i_ytrue = i[1]
        d = manhattan_distance(i_ytrue, i_ypred)
        r = d / float(len(i_ypred))
        y = 1.0
        if r < 0.5:
            y = 0
        else:
            y = 1
        total += ((1 - y) * (2.0 / q) * (e_w**2)) + (y * q * tf.exp(-(2.77 / q) * e_w))

    return total / float(y_pred.shape[0])


def keras_sqrt_diff(tensors):
    t1 = tensors[0]
    t2 = tensors[1]
    # for i in range(1, len(t1)):
    #    s += X[i]
    # s = tf.keras.ops.sqrt(tf.keras.ops.square(s) + 1e-7)
    return tf.keras.ops.abs(t1 - t2)
