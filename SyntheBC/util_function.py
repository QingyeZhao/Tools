import numpy as np
import tensorflow as tf
import random


# dataSet1 : init + track + unsafe
def get_data_set1():
    train_l1_1 = np.loadtxt('./train/trace.txt', dtype=np.float32)
    train_l1_2 = np.loadtxt('./train/init.txt', dtype=np.float32)
    train_l1 = np.concatenate((train_l1_1, train_l1_2), axis=0)
    train_l2 = np.loadtxt('./train/unsafe.txt', dtype=np.float32)

    if train_l1.shape[0] >= 2 * train_l2.shape[0]:
        train_l2 = train_l2.repeat(int(train_l1.shape[0] / train_l2.shape[0]), axis=0)
    elif train_l2.shape[0] >= 2 * train_l1.shape[0]:
        train_l1 = train_l1.repeat(int(train_l2.shape[0] / train_l1.shape[0]), axis=0)

    np.random.shuffle(train_l1)
    np.random.shuffle(train_l2)

    label_l1 = np.array([1, 0] * len(train_l1)).reshape(-1, 2)
    label_l2 = np.array([0, 1] * len(train_l2)).reshape(-1, 2)

    train = np.concatenate((train_l1, train_l2), axis=0)
    label = np.concatenate((label_l1, label_l2), axis=0)
    shuffle_indices = np.random.permutation(train.shape[0])
    train = train[shuffle_indices]
    label = label[shuffle_indices]

    return train, label


def get_data_set2():
    train = np.loadtxt('./train/inv.txt', dtype=np.float32)
    np.random.shuffle(train)
    return train


def get_test_data():
    data1 = np.loadtxt('./test/trace.txt', dtype=np.float32)
    data2 = np.loadtxt('./test/unsafe.txt', dtype=np.float32)

    np.random.shuffle(data1)
    np.random.shuffle(data2)

    test_label1 = np.array([1, 0] * len(data1)).reshape(-1, 2)
    test_label2 = np.array([0, 1] * len(data2)).reshape(-1, 2)

    test = np.concatenate((data1, data2), axis=0)
    label = np.concatenate((test_label1, test_label2), axis=0)

    shuffle_indices = np.random.permutation(test.shape[0])
    test = test[shuffle_indices]
    label = label[shuffle_indices]

    return test, label


# TODO:
def testing(w1, w2, w3, b1, b2, b3):
    input_dim = 2
    invariant_min = [-2, -2]
    invariant_max = [2, 2]

    x = tf.placeholder(tf.float32, [None, 2])
    w1 = tf.Variable(w1)
    b1 = tf.Variable(b1)
    w2 = tf.Variable(w2)
    b2 = tf.Variable(b2)
    w3 = tf.Variable(w3)
    b3 = tf.Variable(b3)

    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
    logits = tf.matmul(hidden2, w3) + b3

    B = logits[:, 0] - logits[:, 1]
    derivation = tf.gradients(B, x)
    x1 = tf.reshape(x[:, 0], [-1, 1])
    x2 = tf.reshape(x[:, 1], [-1, 1])

    x1_derivation = -x1 + 2 * x1 * x1 * x1 * x2 * x2
    x2_derivation = -x2
    part_derivation = tf.concat((x1_derivation, x2_derivation), axis=1)

    derivation_B_tmp = tf.multiply(derivation[0], part_derivation)
    derivation_B = tf.matmul(derivation_B_tmp, np.array([1., 1.], dtype=np.float32).reshape([-1, 1]))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    failed_num = 0
    test_num = 200000

    for i in range(test_num):
        test_points = [0, 0]
        if i % 10000 == 0:
            print(i)
        for i_dim in range(0, input_dim):
            test_points[i_dim] = random.uniform(invariant_min[i_dim], invariant_max[i_dim])

        test_points = np.array(test_points).reshape(-1, 2)

        test_classify = sess.run(B, feed_dict={x: test_points})

        if 0 <= test_classify[0] < 0.005:
            test_result = sess.run(derivation_B, feed_dict={x: test_points})
            if test_result < 0:
                failed_num += 1
                print(failed_num)

    if failed_num > 0:
        print("random points check : %d in %d  -- Fail\n" % (failed_num, test_num))
    else:
        print("random points check : %d -- All Pass\n" % test_num)

