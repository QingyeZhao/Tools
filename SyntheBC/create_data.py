import random
import os


# TODO:
def update(x):
    x1 = x[0]
    x2 = x[1]

    dx1 = -x1 + 2*pow(x1, 3)*pow(x2, 2)
    dx2 = -x2
    result = [x1 + diff_step_length * dx1, x2 + diff_step_length * dx2]

    return result


def simulation(f, x):
    count = 0
    while count < diff_step_count:
        for i_dim in range(0, input_dim):
            if x[i_dim] < invariant_min[i_dim] or x[i_dim] > invariant_max[i_dim]:
                return
        x = update(x)
        count = count + 1
        for i_dim in range(0, input_dim):
            f.write(str(x[i_dim]) + " ")
        f.write("\n")


# TODO:
def create_train_init_data():
    # create initial set data, grid and random
    with open("./train/init.txt", "w") as f:
        piece = [200, 200]
        step_size = [0, 0]
        for i_dim in range(0, input_dim):
            step_size[i_dim] = (initial_max[i_dim] - initial_min[i_dim])/piece[i_dim]
        x = [0, 0]
        for i in range(piece[0] + 1):
            x[0] = initial_min[0] + i * step_size[0]
            for j in range(piece[1] + 1):
                x[1] = initial_min[1] + j * step_size[1]
                for i_dim in range(0, input_dim):
                    f.write(str(x[i_dim]) + " ")
                f.write("\n")

        random_points_num = 100000
        for i in range(0, random_points_num):
            for i_dim in range(0, input_dim):
                x[i_dim] = random.uniform(initial_min[i_dim], initial_max[i_dim])
            for i_dim in range(0, input_dim):
                f.write(str(x[i_dim]) + " ")
            f.write("\n")


# TODO:
def create_train_trace_data():
    # create trace data, grid and random
    with open("./train/trace.txt", "w") as f:
        piece = [50, 50]
        step_size = [0, 0]
        for i_dim in range(0, input_dim):
            step_size[i_dim] = (initial_max[i_dim] - initial_min[i_dim]) / piece[i_dim]
        x = [0, 0]
        for i in range(piece[0] + 1):
            x[0] = initial_min[0] + i * step_size[0]
            for j in range(piece[1] + 1):
                x[1] = initial_min[1] + j * step_size[1]
                simulation(f, x)

        random_points_num = 1000
        for i in range(0, random_points_num):
            for i_dim in range(0, input_dim):
                x[i_dim] = random.uniform(initial_min[i_dim], initial_max[i_dim])
            simulation(f, x)


# TODO:
def create_train_unsafe_data():
    # create unsafe region data, grid and random
    with open("./train/unsafe.txt", "w") as f:
        piece = [200, 200]
        step_size = [0, 0]
        for i_dim in range(0, input_dim):
            step_size[i_dim] = (unsafe_max[i_dim] - unsafe_min[i_dim])/piece[i_dim]
        x = [0, 0]
        for i in range(piece[0] + 1):
            x[0] = invariant_min[0] + i * step_size[0]
            for j in range(piece[1] + 1):
                x[1] = invariant_min[1] + j * step_size[1]
                for i_dim in range(0, input_dim):
                    f.write(str(x[i_dim]) + " ")
                f.write("\n")

        random_points_num = 20000
        for i in range(0, random_points_num):
            for i_dim in range(0, input_dim):
                x[i_dim] = random.uniform(unsafe_min[i_dim], unsafe_max[i_dim])
            for i_dim in range(0, input_dim):
                f.write(str(x[i_dim]) + " ")
            f.write("\n")


# TODO:
def create_train_invariant_data():
    # create invariant data, grid and random
    with open("./train/inv.txt", "w") as f:
        piece = [400, 400]
        step_size = [0, 0]
        for i_dim in range(0, input_dim):
            step_size[i_dim] = (invariant_max[i_dim] - invariant_min[i_dim]) / piece[i_dim]
        x = [0, 0]
        for i in range(piece[0] + 1):
            x[0] = unsafe_min[0] + i * step_size[0]
            for j in range(piece[1] + 1):
                x[1] = unsafe_min[1] + j * step_size[1]
                for i_dim in range(0, input_dim):
                    f.write(str(x[i_dim]) + " ")
                f.write("\n")

        random_points_num = 200000
        for i in range(0, random_points_num):
            for i_dim in range(0, input_dim):
                x[i_dim] = random.uniform(invariant_min[i_dim], invariant_max[i_dim])
            for i_dim in range(0, input_dim):
                f.write(str(x[i_dim]) + " ")
            f.write("\n")


def create_test_trace_data():
    # random points testing trace
    with open("./test/trace.txt", "w") as f:
        x = [0, 0]
        random_points_num = 2000
        for i in range(0, random_points_num):
            for i_dim in range(0, input_dim):
                x[i_dim] = random.uniform(initial_min[i_dim], initial_max[i_dim])
            for i_dim in range(0, input_dim):
                f.write(str(x[i_dim]) + " ")
            f.write("\n")
            simulation(f, x)


def create_test_unsafe_data():
    # random points testing for unsafe region
    with open("./test/unsafe.txt", "w") as f:
        x = [0, 0]
        random_points_num = 2000
        for i in range(0, random_points_num):
            for i_dim in range(0, input_dim):
                x[i_dim] = random.uniform(unsafe_min[i_dim], unsafe_max[i_dim])
            for i_dim in range(0, input_dim):
                f.write(str(x[i_dim]) + " ")
            f.write("\n")

if __name__ == "__main__":
    # TODO:
    input_dim = 2

    invariant_min = [-2, -2]
    invariant_max = [2, 2]

    unsafe_min = [-2, -2]
    unsafe_max = [-1, -1]

    initial_min = [-0.2, 0.3]
    initial_max = [0.2, 0.7]

    diff_step_count = 50
    diff_step_length = 0.001

    if not os.path.exists('train/'):
        os.mkdir('train/')
    if not os.path.exists('test/'):
        os.mkdir('test/')

    create_train_init_data()
    create_train_unsafe_data()
    create_train_trace_data()
    create_train_invariant_data()
    create_test_trace_data()
    create_test_unsafe_data()


