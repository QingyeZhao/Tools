from train.generate_data import generate_data
from train.train_barrier import synthesizing
from verify.verify_barrier import verification
import time
from parameters import *


if __name__ == '__main__':

    only_verification = True
    load_barrier = False
    if only_verification:
        time_start = time.time()
        success_verification_flag, _, _ = verification()
        print("Is the barrier in \"" + load_net_dir + "\" a real barrier? " + str(success_verification_flag))
        time_end = time.time()
        print("Verification time cost:  " + str(time_end - time_start))
    else:
        time_start = time.time()
        # generate_data()
        synthesizing(load_barrier)
        time_end = time.time()
        print("Time cost:  " + str(time_end - time_start))


























