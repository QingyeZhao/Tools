import numpy as np
import gurobipy as gp
from gurobipy import GRB
from verify.interval_number import IntervalNumber, interval_max, inf, sup
from verify.compute_output import get_layer_outputs
from parameters import system_dim
import math
from parameters import number_close_0, number_close_inf, a_bound
from verify.verify_utils import mycallback


def derivative_verify(x_min, x_max, barrier_w, b, controller_w, controller_b):

    barrier_weight_size = len(barrier_w)
    controller_weight_size = len(controller_w)
    input_size = len(barrier_w[0])
    intval_x = []
    for i in range(0, input_size):
        t = IntervalNumber(x_min[i], x_max[i])
        intval_x.append(t)
    intval_x = np.array(intval_x).squeeze().reshape([-1, input_size])
    layer_outputs = get_layer_outputs(intval_x, barrier_w, b)
    barrier_hidden_variable_num = 0
    for i in range(1, barrier_weight_size):
        barrier_hidden_variable_num = barrier_hidden_variable_num + len(barrier_w[i])

    controller_hidden_variable_num = 0
    for i in range(1, controller_weight_size):
        controller_hidden_variable_num = controller_hidden_variable_num + len(controller_w[i])

    intermediate_variable_num = 2  # 3*x1, cos(3*x1)
    all_variable_num = input_size + 3 * barrier_hidden_variable_num + 1 + input_size + intermediate_variable_num + 3 * controller_hidden_variable_num + 10
    lp_A_b_t_row = 3 * barrier_hidden_variable_num + 3 * controller_hidden_variable_num
    lp_Aeq_beq_t_row = barrier_hidden_variable_num + 1 + controller_hidden_variable_num + 5  # 1 (barrier output), 5 (controller output)
    lp_A_Aeq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, all_variable_num))
    lp_b_beq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, 1)).squeeze()
    lp_A_b_loc = -1
    lp_Aeq_beq_loc = lp_A_b_t_row - 1

    binary = np.zeros((barrier_hidden_variable_num + controller_hidden_variable_num, 1), dtype=np.int).squeeze()
    binary_loc = -1

    lp_lb = np.zeros((all_variable_num, 1)).squeeze() - number_close_inf
    lp_ub = np.ones((all_variable_num, 1)).squeeze() + number_close_inf

    # ##########################################################################################
    # encode barrier
    # ##########################################################################################
    for i in range(0, input_size):
        lp_lb[i] = inf(intval_x[0][i])
        lp_ub[i] = sup(intval_x[0][i])

    lp_lb_ub_loc = input_size - 1
    process_location = input_size - 1
    y_head_record = np.zeros((1, input_size), dtype=np.int).squeeze()
    for i in range(0, input_size):
        y_head_record[i] = i

    for layer_index in range(0, barrier_weight_size - 1):
        layer_output_before_relu = layer_outputs[layer_index]
        y_head_record_last_layer = y_head_record
        y_head_record = np.zeros((1, len(barrier_w[layer_index][0])), dtype=np.int).squeeze()
        for j in range(0, len(barrier_w[layer_index][0])):
            lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
            lp_lb[lp_lb_ub_loc + 2] = max(0, inf(layer_output_before_relu[0][j]))
            lp_lb[lp_lb_ub_loc + 3] = 0
            lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
            lp_ub[lp_lb_ub_loc + 2] = max(0, sup(layer_output_before_relu[0][j]))
            lp_ub[lp_lb_ub_loc + 3] = 1
            lp_lb_ub_loc = lp_lb_ub_loc + 3

            binary[binary_loc + 1] = process_location + 3
            binary_loc = binary_loc + 1

            lp_A_t = np.zeros((1, all_variable_num))

            lp_A_t[0][process_location + 1] = -1
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -inf(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 1] = -inf(layer_output_before_relu[0][j])

            lp_A_t[0][process_location + 1] = 1
            lp_A_t[0][process_location + 2] = -1
            lp_A_t[0][process_location + 3] = 0
            lp_A_Aeq[lp_A_b_loc + 2, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 2] = 0

            lp_A_t[0][process_location + 1] = 0
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -sup(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 3, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 3] = 0
            lp_A_b_loc = lp_A_b_loc + 3

            lp_Aeq_t = np.zeros((1, all_variable_num))
            for k in range(0, len(barrier_w[layer_index])):
                lp_Aeq_t[0][y_head_record_last_layer[k]] = barrier_w[layer_index][k, j]
            lp_Aeq_t[0][process_location + 1] = -1
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1

            y_head_record[j] = process_location + 2
            process_location = process_location + 3
            layer_output_before_relu[0][j] = interval_max(layer_output_before_relu[0][j], 0)

    layer_index = barrier_weight_size - 1
    layer_output_before_relu = layer_outputs[layer_index]
    y_head_record_last_layer = y_head_record
    t_process_loc = process_location

    for j in range(0, len(barrier_w[layer_index][0])):  # len(barrier_w[layer_index][0])==1 forever
        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(barrier_w[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = barrier_w[layer_index][k, j]

        lp_Aeq_t[0][t_process_loc + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        # lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
        # lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
        lp_lb[lp_lb_ub_loc + 1] = 0
        lp_ub[lp_lb_ub_loc + 1] = 0
        lp_lb_ub_loc = lp_lb_ub_loc + 1
        t_process_loc = t_process_loc + 1
    process_location = process_location + len(barrier_w[layer_index][0])

    vtype = list("C" * all_variable_num)


    m = gp.Model()

    all_vars = []
    for i in range(0, all_variable_num):
        all_vars.append(m.addVar(lb=lp_lb[i], ub=lp_ub[i], vtype=vtype[i]))


    # ##########################################################################################
    # encode controller
    # ##########################################################################################

    controller_process_location = process_location + input_size + intermediate_variable_num
    lp_lb_ub_loc = controller_process_location
    layer_outputs = get_layer_outputs(intval_x, controller_w, controller_b)
    y_head_record = np.zeros((1, input_size), dtype=np.int).squeeze()
    for i in range(0, input_size):
        y_head_record[i] = i

    for layer_index in range(0, controller_weight_size - 1):
        layer_output_before_relu = layer_outputs[layer_index]
        y_head_record_last_layer = y_head_record
        y_head_record = np.zeros((1, len(controller_w[layer_index][0])), dtype=np.int).squeeze()
        for j in range(0, len(controller_w[layer_index][0])):
            lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
            lp_lb[lp_lb_ub_loc + 2] = max(0, inf(layer_output_before_relu[0][j]))
            lp_lb[lp_lb_ub_loc + 3] = 0
            lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
            lp_ub[lp_lb_ub_loc + 2] = max(0, sup(layer_output_before_relu[0][j]))
            lp_ub[lp_lb_ub_loc + 3] = 1
            lp_lb_ub_loc = lp_lb_ub_loc + 3

            binary[binary_loc + 1] = controller_process_location + 3
            binary_loc = binary_loc + 1

            lp_A_t = np.zeros((1, all_variable_num))

            lp_A_t[0][controller_process_location + 1] = -1
            lp_A_t[0][controller_process_location + 2] = 1
            lp_A_t[0][controller_process_location + 3] = -inf(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 1] = -inf(layer_output_before_relu[0][j])

            lp_A_t[0][controller_process_location + 1] = 1
            lp_A_t[0][controller_process_location + 2] = -1
            lp_A_t[0][controller_process_location + 3] = 0
            lp_A_Aeq[lp_A_b_loc + 2, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 2] = 0

            lp_A_t[0][controller_process_location + 1] = 0
            lp_A_t[0][controller_process_location + 2] = 1
            lp_A_t[0][controller_process_location + 3] = -sup(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 3, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 3] = 0
            lp_A_b_loc = lp_A_b_loc + 3

            lp_Aeq_t = np.zeros((1, all_variable_num))
            for k in range(0, len(controller_w[layer_index])):
                lp_Aeq_t[0][y_head_record_last_layer[k]] = controller_w[layer_index][k, j]
            lp_Aeq_t[0][controller_process_location + 1] = -1
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = -controller_b[layer_index][j]
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1

            y_head_record[j] = controller_process_location + 2
            controller_process_location = controller_process_location + 3
            layer_output_before_relu[0][j] = interval_max(layer_output_before_relu[0][j], 0)


    layer_index=2
    layer_output_before_relu = layer_outputs[layer_index]
    y_head_record_last_layer = y_head_record
    y_head_record = np.zeros((1, len(controller_w[layer_index][0])), dtype=np.int).reshape([1])
    for j in range(0, len(controller_w[layer_index][0])):

        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(controller_w[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = controller_w[layer_index][k, j]
        lp_Aeq_t[0][controller_process_location + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -controller_b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
        lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])

        # t2=-t1
        lp_Aeq_t = np.zeros((1, all_variable_num))
        lp_Aeq_t[0][controller_process_location + 1] = 1
        lp_Aeq_t[0][controller_process_location + 2] = 1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = 0
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        lp_lb[lp_lb_ub_loc + 2] = -lp_ub[lp_lb_ub_loc + 1]
        lp_ub[lp_lb_ub_loc + 2] = -lp_lb[lp_lb_ub_loc + 1]

        # t3=exp(t1)
        xt = all_vars[controller_process_location + 1]
        yt = all_vars[controller_process_location + 3]
        m.addGenConstrExp(xvar=xt, yvar=yt)
        lp_lb[lp_lb_ub_loc + 3] = math.exp(lp_lb[lp_lb_ub_loc + 1])
        lp_ub[lp_lb_ub_loc + 3] = math.exp(lp_ub[lp_lb_ub_loc + 1])

        # t4=exp(t2)
        xt = all_vars[controller_process_location + 2]
        yt = all_vars[controller_process_location + 4]
        m.addGenConstrExp(xvar=xt, yvar=yt)
        lp_lb[lp_lb_ub_loc + 4] = math.exp(lp_lb[lp_lb_ub_loc + 2])
        lp_ub[lp_lb_ub_loc + 4] = math.exp(lp_ub[lp_lb_ub_loc + 2])

        # t5=t3-t4
        lp_Aeq_t = np.zeros((1, all_variable_num))
        lp_Aeq_t[0][controller_process_location + 3] = 1
        lp_Aeq_t[0][controller_process_location + 4] = -1
        lp_Aeq_t[0][controller_process_location + 5] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = 0
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        lp_lb[lp_lb_ub_loc + 5] = lp_lb[lp_lb_ub_loc + 3] - lp_ub[lp_lb_ub_loc + 4]
        lp_ub[lp_lb_ub_loc + 5] = lp_ub[lp_lb_ub_loc + 3] - lp_lb[lp_lb_ub_loc + 4]

        # t6=t3+t4
        lp_Aeq_t = np.zeros((1, all_variable_num))
        lp_Aeq_t[0][controller_process_location + 3] = 1
        lp_Aeq_t[0][controller_process_location + 4] = 1
        lp_Aeq_t[0][controller_process_location + 6] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = 0
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        lp_lb[lp_lb_ub_loc + 6] = lp_lb[lp_lb_ub_loc + 3] + lp_lb[lp_lb_ub_loc + 4]
        lp_ub[lp_lb_ub_loc + 6] = lp_ub[lp_lb_ub_loc + 3] + lp_ub[lp_lb_ub_loc + 4]

        # t7=log(t6)
        xt = all_vars[controller_process_location + 6]
        yt = all_vars[controller_process_location + 7]
        m.addGenConstrLog(xvar=xt, yvar=yt)
        lp_lb[lp_lb_ub_loc + 7] = math.log(lp_lb[lp_lb_ub_loc + 6])
        lp_ub[lp_lb_ub_loc + 7] = math.log(lp_ub[lp_lb_ub_loc + 6])

        # t8=-t7
        lp_Aeq_t = np.zeros((1, all_variable_num))
        lp_Aeq_t[0][controller_process_location + 7] = 1
        lp_Aeq_t[0][controller_process_location + 8] = 1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = 0
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        lp_lb[lp_lb_ub_loc + 8] = -lp_ub[lp_lb_ub_loc + 7]
        lp_ub[lp_lb_ub_loc + 8] = -lp_lb[lp_lb_ub_loc + 7]

        # t9=exp(t8)
        xt = all_vars[controller_process_location + 8]
        yt = all_vars[controller_process_location + 9]
        m.addGenConstrExp(xvar=xt, yvar=yt)
        lp_lb[lp_lb_ub_loc + 9] = math.exp(lp_lb[lp_lb_ub_loc + 8])
        lp_ub[lp_lb_ub_loc + 9] = math.exp(lp_ub[lp_lb_ub_loc + 8])

        # t10=t5*t9
        lp_q1 = np.zeros((all_variable_num, all_variable_num))
        lp_q1[controller_process_location + 5][controller_process_location + 9] = 1
        c_t = np.zeros((1, all_variable_num)).squeeze()
        c_t[controller_process_location + 10] = -1
        m.addMQConstr(Q=lp_q1, c=c_t, sense='=', rhs=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars)
        lp_lb[controller_process_location + 10] = -number_close_inf
        lp_ub[controller_process_location + 10] = number_close_inf

        y_head_record[j] = controller_process_location + 10
        controller_process_location = controller_process_location + 10
        lp_lb_ub_loc = lp_lb_ub_loc + 10

    for i_c in range(0, len(binary)):
        tvar = all_vars[binary[i_c]]
        tvar.setAttr("vtype", 'B')


    m.addMConstrs(A=lp_A_Aeq[0:lp_A_b_t_row], x=all_vars, sense='<', b=lp_b_beq[0:lp_A_b_t_row])
    m.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row:], x=all_vars, sense='=', b=lp_b_beq[lp_A_b_t_row:])


    lp_Aeq_d1 = np.zeros((1, all_variable_num))
    lp_Aeq_d2 = np.zeros((1, all_variable_num))
    k = np.array(barrier_w[1]).reshape([1, -1])
    for i in range(0, len(barrier_w[0][0])):
        lp_Aeq_d1[0][binary[i]] = barrier_w[0][0, i] * k[0][i]
    lp_Aeq_d1[0][process_location + 1] = -1
    for i in range(0, len(barrier_w[0][0])):
        lp_Aeq_d2[0][binary[i]] = barrier_w[0][1, i] * k[0][i]
    lp_Aeq_d2[0][process_location + 2] = -1
    lp_Aeq_derivative = np.concatenate((lp_Aeq_d1, lp_Aeq_d2), axis=0)
    lp_beq_derivative = np.zeros((2, 1)).squeeze()
    m.addMConstrs(A=lp_Aeq_derivative, x=all_vars, sense='=', b=lp_beq_derivative)

    lp_lb[process_location + 1] = -number_close_inf
    lp_ub[process_location + 1] = number_close_inf
    lp_lb[process_location + 2] = -number_close_inf
    lp_ub[process_location + 2] = number_close_inf

    # 3*x1
    lp_Aeq_t = np.zeros((1, all_variable_num))
    lp_Aeq_t[0][0] = 3
    lp_Aeq_t[0][process_location + 3] = -1
    lp_beq_t = np.zeros((1, 1)).reshape([1])
    m.addMConstrs(A=lp_Aeq_t, x=all_vars, sense='=', b=lp_beq_t)

    lp_lb[process_location + 3] = lp_lb[0]*3
    lp_ub[process_location + 3] = lp_ub[0]*3

    # cos(3*x1)
    xt = all_vars[process_location + 3]
    yt = all_vars[process_location + 4]
    m.addGenConstrSin(xvar=xt, yvar=yt)
    lp_lb[process_location + 4] = -1
    lp_ub[process_location + 4] = 1

    for x_i in range(0, all_variable_num):
        tvar = all_vars[x_i]
        tvar.setAttr("lb", lp_lb[x_i])
        tvar.setAttr("ub", lp_ub[x_i])

    lp_f = np.zeros((1, all_variable_num)).squeeze()
    # quadratic objective
    q_new = np.zeros((all_variable_num, all_variable_num))
    q_new[process_location + 1, 1] = 1
    q_new[process_location + 2, process_location + 4] = -0.0025
    q_new[process_location + 2, all_variable_num-1] = a_bound * 0.0015

    m.setMObjective(Q=q_new, c=lp_f, constant=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars, sense=GRB.MAXIMIZE)

    m.setParam('OutputFlag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('FuncPieces', -1)
    # m.setParam("FuncPieceError", 0.001)

    m.optimize(mycallback)
    # if m.Status == 3:  # can not found optima, approximation is too loose
    #     return -1, 0, 0
    c1 = m.objVal
    c1_gap = m.MIPGap
    c1_v = m.getVars()
    c1_x = []
    for i in range(0, system_dim):
        c1_x.append(c1_v[i].x)
    return c1, c1_gap, c1_x









