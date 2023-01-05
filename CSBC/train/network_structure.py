import tensorflow as tf
from parameters import *


def network_parameter_random(layer_neurons, num_hidden_layers, trainable_flag=True):
    w1 = tf.Variable(tf.truncated_normal((layer_neurons[0], layer_neurons[1]), 0.1), trainable=trainable_flag)
    b1 = tf.Variable(tf.zeros([layer_neurons[1]]), trainable=trainable_flag)
    w2 = tf.Variable(tf.truncated_normal((layer_neurons[1], layer_neurons[2]), 0.1), trainable=trainable_flag)
    b2 = tf.Variable(tf.zeros([layer_neurons[2]]))
    if num_hidden_layers == 1:
        return [w1, b1, w2, b2]
    else:
        print("Not implement!\n")
        exit(-1)


def network_parameter_load_controller(trainable_flag=False):
    w1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "a/w1", dtype=np.float32)), trainable=trainable_flag)
    b1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "a/b1", dtype=np.float32)), trainable=trainable_flag)
    w2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "a/w2", dtype=np.float32)), trainable=trainable_flag)
    b2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "a/b2", dtype=np.float32)), trainable=trainable_flag)
    w3 = tf.Variable(
        tf.constant(np.loadtxt(load_net_dir + "a/w3", dtype=np.float32).reshape([-1, 1])), trainable=trainable_flag)
    b3 = tf.Variable(
        tf.constant(np.loadtxt(load_net_dir + "a/b3", dtype=np.float32).reshape([1])), trainable=trainable_flag)
    return [w1, b1, w2, b2, w3, b3]


def network_parameter_load(sub_net_dir_name, num_hidden_layers, trainable_flag=True):
    w1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w1", dtype=np.float32)), trainable=trainable_flag)
    b1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b1", dtype=np.float32)), trainable=trainable_flag)
    if num_hidden_layers == 1:
        w2 = tf.Variable(
            tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w2", dtype=np.float32).reshape([-1, 1])), trainable=trainable_flag)
        b2 = tf.Variable(
            tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b2", dtype=np.float32).reshape([1])), trainable=trainable_flag)
        return [w1, b1, w2, b2]
    else:
        w2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w2", dtype=np.float32)), trainable=trainable_flag)
        b2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b2", dtype=np.float32)), trainable=trainable_flag)
        if num_hidden_layers == 2:
            w3 = tf.Variable(
                tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w3", dtype=np.float32).reshape([-1, 1])), trainable=trainable_flag)
            b3 = tf.Variable(
                tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b3", dtype=np.float32).reshape([1])), trainable=trainable_flag)
            return [w1, b1, w2, b2, w3, b3]
        else:
            w3 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w3", dtype=np.float32)), trainable=trainable_flag)
            b3 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b3", dtype=np.float32)), trainable=trainable_flag)
            if num_hidden_layers == 3:
                w4 = tf.Variable(
                    tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w4", dtype=np.float32).reshape([-1, 1])), trainable=trainable_flag)
                b4 = tf.Variable(
                    tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b4", dtype=np.float32).reshape([1])), trainable=trainable_flag)
                return [w1, b1, w2, b2, w3, b3, w4, b4]
            else:
                w4 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w4", dtype=np.float32)), trainable=trainable_flag)
                b4 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b4", dtype=np.float32)), trainable=trainable_flag)
                if num_hidden_layers == 4:
                    w5 = tf.Variable(
                        tf.constant(
                            np.loadtxt(load_net_dir + sub_net_dir_name + "/w5", dtype=np.float32).reshape([-1, 1])), trainable=trainable_flag)
                    b5 = tf.Variable(
                        tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b5", dtype=np.float32).reshape([1])), trainable=trainable_flag)
                    return [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]
                else:
                    w5 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w5", dtype=np.float32)), trainable=trainable_flag)
                    b5 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b5", dtype=np.float32)), trainable=trainable_flag)
                    if num_hidden_layers == 5:
                        w6 = tf.Variable(
                            tf.constant(
                                np.loadtxt(load_net_dir + sub_net_dir_name + "/w6", dtype=np.float32).reshape([-1, 1])), trainable=trainable_flag)
                        b6 = tf.Variable(
                            tf.constant(
                                np.loadtxt(load_net_dir + sub_net_dir_name + "/b6", dtype=np.float32).reshape([1])), trainable=trainable_flag)
                        return [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6]
                    else:
                        w6 = tf.Variable(
                            tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/w6", dtype=np.float32)), trainable=trainable_flag)
                        b6 = tf.Variable(
                            tf.constant(np.loadtxt(load_net_dir + sub_net_dir_name + "/b6", dtype=np.float32)), trainable=trainable_flag)
                        if num_hidden_layers == 6:
                            w7 = tf.Variable(
                                tf.constant(
                                    np.loadtxt(load_net_dir + sub_net_dir_name + "/w7", dtype=np.float32).reshape(
                                        [-1, 1])), trainable=trainable_flag)
                            b7 = tf.Variable(
                                tf.constant(
                                    np.loadtxt(load_net_dir + sub_net_dir_name + "/b7", dtype=np.float32).reshape([1])), trainable=trainable_flag)
                            return [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7]
                        else:
                            print("Not implement!\n")
                            exit(-1)





























