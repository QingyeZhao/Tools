import numpy as np
import tensorflow as tf
import os

from util_function import get_test_data, get_data_set1, get_data_set2, testing


def compute_accuracy(v_xs, v_ys, logits, x, sess):
    y_pre = sess.run(logits, feed_dict={x: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs})
    return result


def weights_bias(n_inputs, n_neurons):
    stddev = 2 / np.sqrt(n_inputs)
    init = tf.truncated_normal((n_inputs, n_neurons), stddev)
    return tf.Variable(init), tf.Variable(tf.zeros([n_neurons]))


def train_barrier():
    # structure of the network
    layer = 4
    n_hidden = [2, 20, 16, 2]

    # get data
    data_set_1, label = get_data_set1()
    data_set_2 = get_data_set2()
    test_data, test_label = get_test_data()

    # define the network
    input1 = tf.placeholder(tf.float32, [None, 2], name="data_set_1")
    y = tf.placeholder(tf.float32, [None, 2])

    w1, b1 = weights_bias(n_hidden[0], n_hidden[1])
    w2, b2 = weights_bias(n_hidden[1], n_hidden[2])
    w3, b3 = weights_bias(n_hidden[2], n_hidden[3])

    h1 = tf.nn.relu(tf.matmul(input1, w1) + b1, name="layer1")
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2, name="layer2")
    h3 = tf.matmul(h2, w3) + b3

    # define loss1
    L1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=h3, labels=y))
    logit = tf.nn.sigmoid(h3, name="logit_output")

    # to get the derivation of all invariant points
    input2 = tf.placeholder(tf.float32, [None, 2], name="data_set_2")
    hidden1 = tf.nn.relu(tf.matmul(input2, w1) + b1, name="der_layer1")
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2, name="der_layer2")
    hidden3 = tf.matmul(hidden2, w3) + b3

    # define loss2
    # first : define part derivation
    x1 = tf.reshape(input2[:, 0], [-1, 1])
    x2 = tf.reshape(input2[:, 1], [-1, 1])

    # TODO:
    x1_derivation = -x1 + 2*x1*x1*x1*x2*x2
    x2_derivation = -x2

    part_derivation = tf.concat((x1_derivation, x2_derivation), axis=1)

    # second : compute the gradient of B = y1-y2
    B = tf.reshape(hidden3[:, 0], [-1, 1]) - tf.reshape(hidden3[:, 1], [-1, 1])
    cond_B1 = tf.cast(tf.greater_equal(B, 0), tf.float32)
    cond_B2 = tf.cast(tf.less_equal(B, 0.005), tf.float32)

    gradient = tf.gradients(B, input2)
    derivation_B_tmp = tf.multiply(gradient[0], part_derivation)
    derivation_B = tf.matmul(derivation_B_tmp, np.array([1., 1.], dtype=np.float32).reshape([-1, 1]))
    derivation_B = tf.multiply(tf.multiply(derivation_B, cond_B1), cond_B2)

    L2 = tf.reduce_sum(tf.abs(tf.minimum(derivation_B, 0.00)))
    # train
    epoch_num = 2
    batch_size1 = 1280
    batch_size2 = 1280

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    alpha = 0.5
    loss = alpha * L1 + (1 - alpha) * L2
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    sess.run(tf.initialize_all_variables())

    stop_flag = 0

    for i in range(epoch_num):
        batch_num1 = int(data_set_1.shape[0] / batch_size1)
        batch_num2 = int(data_set_2.shape[0] / batch_size2)

        for j in range(batch_num1):
            batch_data1 = data_set_1[batch_size1 * j:batch_size1 * (j + 1), :]
            batch_label1 = label[batch_size1 * j:batch_size1 * (j + 1), :]

            for k in range(batch_num2):
                batch_data2 = data_set_2[batch_size2 * k:batch_size2 * (k + 1), :]
                sess.run(train_step, feed_dict={input1: batch_data1, y: batch_label1, input2: batch_data2})

            if j % 10 == 0:
                print(j)
                epoch_test_accuracy = compute_accuracy(test_data, test_label, logit, input1, sess)
                epoch_train_accuracy = compute_accuracy(data_set_1, label, logit, input1, sess)
                epoch_train_loss = sess.run(loss, feed_dict={input1: data_set_1, y: label, input2: data_set_2})
                epoch_test_loss = sess.run(loss, feed_dict={input1: test_data, y: test_label, input2: data_set_2})
                epoch_L2 = sess.run(L2, feed_dict={input2: data_set_2})
                epoch_L1 = sess.run(L1, feed_dict={input1: data_set_1, y: label})

                print("***epoch %d | train accuracy : %f test accuracy : %f\n" % (
                    i, epoch_train_accuracy, epoch_test_accuracy))
                print("***train loss : %f test loss : %f L1 : %f L2 : %f\n" % (
                    epoch_train_loss, epoch_test_loss, epoch_L1, epoch_L2))

                if epoch_train_accuracy == 1 and epoch_test_accuracy == 1 and epoch_L2 == 0:
                    stop_flag = 1
                    break
        if stop_flag == 1:
            break

    w1 = sess.run(w1)
    b1 = sess.run(b1)
    w2 = sess.run(w2)
    b2 = sess.run(b2)
    w3 = sess.run(w3)
    b3 = sess.run(b3)
    sess.close()

    # save parameters of the net
    if not os.path.exists('net/'):
        os.mkdir('net/')
    np.savetxt('net/w1', w1)
    np.savetxt('net/w2', w2)
    np.savetxt('net/w3', w3)
    np.savetxt('net/b1', b1)
    np.savetxt('net/b2', b2)
    np.savetxt('net/b3', b3)

    with open("./net/structure.txt", "w") as f:
        f.write(str(layer) + "\n")
        for i in range(0, layer):
            f.write(str(n_hidden[i]) + "\n")


def test_barrier():
    w1 = np.loadtxt('net/w1', dtype=np.float32)
    w2 = np.loadtxt('net/w2', dtype=np.float32)
    w3 = np.loadtxt('net/w3', dtype=np.float32)
    b1 = np.loadtxt('net/b1', dtype=np.float32)
    b2 = np.loadtxt('net/b2', dtype=np.float32)
    b3 = np.loadtxt('net/b3', dtype=np.float32)
    testing(w1, w2, w3, b1, b2, b3)


if __name__ == "__main__":
    # train_barrier()
    test_barrier()

