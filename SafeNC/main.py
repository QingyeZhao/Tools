from train.generate_data import generate_data
from train.train_controller import synthesizing
from certify.certify_controller import certification
import time
from parameters import *


if __name__ == '__main__':

    only_certification = True
    load_controller = False
    load_barrier = False
    if only_certification:
        time_start = time.time()
        success_certification_flag, _, _ = certification()
        print("Is the controller in \"" + load_net_dir + "\" safe? " + str(success_certification_flag))
        time_end = time.time()
        print("Certification time cost:  " + str(time_end - time_start))
    else:
        time_start = time.time()
        generate_data()
        synthesizing(load_controller, load_barrier)
        time_end = time.time()
        print("Time cost:  " + str(time_end - time_start))




























