from train.network_structure import *
from train.train_utils import *
from certify.certify_controller import certification
from parameters import *


def get_network_output(n_type, network_parameter, x):
    network_layers = int(len(network_parameter)/2)
    for i in range(0, network_layers-1):
        if n_type == 2:
            x = tf.nn.tanh(tf.matmul(x, network_parameter[i*2]) + network_parameter[i*2+1])
        elif n_type == 1:
            x = tf.nn.sigmoid(tf.matmul(x, network_parameter[i*2]) + network_parameter[i*2+1])
        else:
            x = tf.nn.relu(tf.matmul(x, network_parameter[i*2]) + network_parameter[i*2+1])
    x = tf.matmul(x, network_parameter[network_layers*2 - 2]) + network_parameter[network_layers*2 - 1]
    return x


def train_controller(load_controller, load_barrier, save_net_dir):
    if load_controller:
        controller_layer_neurons = []
        t1 = np.loadtxt(load_net_dir + "controller_structure", dtype=np.int).squeeze()
        for i in range(0, len(t1)):
            controller_layer_neurons.append(t1[i])
        controller_type = np.loadtxt(load_net_dir + "controller_type", dtype=np.int).squeeze()
    else:
        controller_layer_neurons = default_controller_layer_neurons
        controller_type = default_controller_type

    if load_barrier:
        barrier_layer_neurons = []
        t1 = np.loadtxt(load_net_dir + "barrier_structure", dtype=np.int).squeeze()
        for i in range(0, len(t1)):
            barrier_layer_neurons.append(t1[i])
        barrier_type = np.loadtxt(load_net_dir + "barrier_type", dtype=np.int).squeeze()
    else:
        barrier_layer_neurons = default_barrier_layer_neurons
        barrier_type = default_barrier_type

    x_unsafe = tf.placeholder(tf.float32, [None, system_dim], name="x_unsafe")
    x_init = tf.placeholder(tf.float32, [None, system_dim], name="x_init")
    x_inv = tf.placeholder(tf.float32, [None, system_dim], name="x_inv")

    if load_controller:
        controller_saving_parameters = network_parameter_load("controller", len(controller_layer_neurons) - 2)
    else:
        controller_saving_parameters = network_parameter_random(controller_layer_neurons, len(controller_layer_neurons) - 2)
    controller_output_inv = get_network_output(controller_type, controller_saving_parameters, x_inv)

    if load_barrier:
        barrier_saving_parameters = network_parameter_load("barrier", len(barrier_layer_neurons) - 2)
    else:
        barrier_saving_parameters = network_parameter_random(barrier_layer_neurons, len(barrier_layer_neurons) - 2)
    barrier_output_unsafe = get_network_output(barrier_type, barrier_saving_parameters, x_unsafe)
    barrier_output_init = get_network_output(barrier_type, barrier_saving_parameters, x_init)
    barrier_output_inv = get_network_output(barrier_type, barrier_saving_parameters, x_inv)

    loss_unsafe = tf.reduce_sum(-tf.minimum(0., barrier_output_unsafe))

    loss_init = tf.reduce_sum(tf.maximum(0., barrier_output_init))

    x1 = tf.reshape(x_inv[:, 0], [-1, 1])
    x2 = tf.reshape(x_inv[:, 1], [-1, 1])
    x1_derivative = tf.sin(x2)
    x2_derivative = -controller_output_inv
    system_derivative = tf.concat((x1_derivative, x2_derivative), axis=1)

    cond1 = tf.cast(tf.greater_equal(barrier_output_inv, -0.01), tf.float32)
    cond2 = tf.cast(tf.less_equal(barrier_output_inv, 0.01), tf.float32)

    barrier_derivative = tf.gradients(barrier_output_inv, x_inv)
    lie_derivative_before_sum = tf.multiply(barrier_derivative[0], system_derivative)
    lie_derivative_sum = tf.matmul(lie_derivative_before_sum, np.array([1.] * system_dim, dtype=np.float32).reshape([-1, 1]))
    lie_derivative = tf.multiply(tf.multiply(lie_derivative_sum, cond1), cond2)

    loss_inv = tf.reduce_sum(tf.maximum(lie_derivative, 0.00))

    for i in range(0, int(len(controller_saving_parameters)/2)):
        tf.add_to_collection("reg_loss", tf.contrib.layers.l1_regularizer(0.5)(controller_saving_parameters[i * 2]))
        tf.add_to_collection("reg_loss", tf.contrib.layers.l2_regularizer(0.5)(controller_saving_parameters[i * 2]))

    for i in range(0, int(len(barrier_saving_parameters)/2)):
        tf.add_to_collection("reg_loss", tf.contrib.layers.l1_regularizer(0.5)(barrier_saving_parameters[i * 2]))
        tf.add_to_collection("reg_loss", tf.contrib.layers.l2_regularizer(0.5)(barrier_saving_parameters[i * 2]))

    loss_reg = tf.get_collection("reg_loss")
    k = 1


    alpha = 1
    beta = 1
    gamma = 1
    loss = alpha * loss_unsafe + beta * loss_init + gamma * loss_inv
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    data_unsafe, data_init, data_inv = get_data()
    data_size = np.array([data_unsafe.shape[0], data_init.shape[0], data_inv.shape[0]], dtype=np.int).squeeze()

    epoch_num = 50
    batch_num = 100
    batch_size = np.array(data_size/batch_num).astype(np.int)
    stop_flag = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    for ep in range(0, epoch_num):
        for i in range(0, batch_num):
            batch_unsafe = data_unsafe[i * batch_size[0]:(i + 1) * batch_size[0], :]
            batch_init = data_init[i * batch_size[1]:(i + 1) * batch_size[1], :]
            batch_inv = data_inv[i * batch_size[2]:(i + 1) * batch_size[2], :]

            if i % 1 == 0:
                epoch_loss = sess.run(loss, feed_dict={x_unsafe: data_unsafe, x_init: data_init, x_inv: data_inv})
                epoch_loss_unsafe = sess.run(loss_unsafe, feed_dict={x_unsafe: data_unsafe})
                epoch_loss_init = sess.run(loss_init, feed_dict={x_init: data_init})
                epoch_loss_inv = sess.run(loss_inv, feed_dict={x_inv: data_inv})

                epoch_loss_reg = sum(sess.run(loss_reg))

                print("epoch: %d/%d, batch: %d/%d, " % (ep, epoch_num-1, i, batch_num-1), end="")
                print("overall loss: %f, loss_unsafe: %f, loss_init: %f, loss_inv: %f, loss_reg: %f" %
                      (epoch_loss, epoch_loss_unsafe, epoch_loss_init, epoch_loss_inv, epoch_loss_reg))

                if epoch_loss == 0:
                    stop_flag = 1
                    break
            sess.run(train_step, feed_dict={x_unsafe: batch_unsafe, x_init: batch_init, x_inv: batch_inv})
        if stop_flag == 1:
            break
    epoch_loss = sess.run(loss, feed_dict={x_unsafe: data_unsafe, x_init: data_init, x_inv: data_inv})
    epoch_loss_reg = sum(sess.run(loss_reg))
    if epoch_loss == 0:
        r = True
    else:
        r = False

    save_parameters(sess, barrier_saving_parameters, barrier_layer_neurons, barrier_type,
                    controller_saving_parameters, controller_layer_neurons, controller_type, save_net_dir)
    sess.close()
    tf.reset_default_graph()

    return r


def synthesizing(load_controller=True, load_barrier=True):
    save_net_dir = load_net_dir
    success_certification_flag = False
    while not success_certification_flag:
        success_training_flag = False
        while not success_training_flag:
            success_training_flag = train_controller(load_controller=load_controller, load_barrier=load_barrier, save_net_dir=save_net_dir)
            load_barrier = True
            load_controller = True
        success_certification_flag, counterexample, dataset_flag = certification()
        if not success_certification_flag:
            extend_counterexample(counterexample, dataset_flag)



