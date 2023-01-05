import numpy as np
import gurobipy as gp
from gurobipy import GRB
from certify.interval_number import IntervalNumber, interval_max, inf, sup
from certify.compute_output import get_layer_outputs, get_layer_outputs_sigmoid
from parameters import system_dim, number_close_inf
import math
from certify.certify_utils import mycallback


def derivative_certify(x_min, x_max, barrier_w, b, controller_w, controller_b):

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

    all_variable_num = input_size + 3 * barrier_hidden_variable_num + 1 + input_size + 1 + 7 * controller_hidden_variable_num + 1
    lp_A_b_t_row = 3 * barrier_hidden_variable_num
    lp_Aeq_beq_t_row = barrier_hidden_variable_num + 4 * controller_hidden_variable_num + 1 + 1
    lp_A_Aeq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, all_variable_num))
    lp_b_beq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, 1)).squeeze()
    lp_A_b_loc = -1
    lp_Aeq_beq_loc = lp_A_b_t_row - 1

    binary = np.zeros((barrier_hidden_variable_num, 1), dtype=np.int).squeeze()
    binary_loc = -1

    lp_lb = np.zeros((all_variable_num, 1)).squeeze()
    lp_ub = np.ones((all_variable_num, 1)).squeeze()

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

    for j in range(0, len(barrier_w[layer_index][0])):
        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(barrier_w[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = barrier_w[layer_index][k, j]

        lp_Aeq_t[0][t_process_loc + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1

        lp_lb[lp_lb_ub_loc + 1] = 0
        lp_ub[lp_lb_ub_loc + 1] = 0
        lp_lb_ub_loc = lp_lb_ub_loc + 1
        t_process_loc = t_process_loc + 1
    process_location = process_location + len(barrier_w[layer_index][0])

    vtype = list("C" * all_variable_num)
    for i_c in range(0, len(binary)):
        vtype[binary[i_c]] = 'B'
    vtype = ''.join(vtype)

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('NonConvex', 2)
    all_vars = []
    for i in range(0, all_variable_num):
        all_vars.append(m.addVar(lb=lp_lb[i], ub=lp_ub[i], vtype=vtype[i]))
    # ##########################################################################################
    # encode controller
    # ##########################################################################################
    controller_process_location = process_location + input_size + 1  # 1:sin(x2)
    lp_lb_ub_loc = controller_process_location
    layer_outputs = get_layer_outputs_sigmoid(intval_x, controller_w, controller_b)
    y_head_record = np.zeros((1, input_size), dtype=np.int).squeeze()
    for i in range(0, input_size):
        y_head_record[i] = i

    for layer_index in range(0, controller_weight_size - 1):
        layer_output_before_relu = layer_outputs[layer_index]
        y_head_record_last_layer = y_head_record
        y_head_record = np.zeros((1, len(controller_w[layer_index][0])), dtype=np.int).squeeze()
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

            # t3=exp(t2)
            xt = all_vars[controller_process_location + 2]
            yt = all_vars[controller_process_location + 3]
            m.addGenConstrExp(xvar=xt, yvar=yt)
            lp_lb[lp_lb_ub_loc + 3] = math.exp(lp_lb[lp_lb_ub_loc + 2])
            lp_ub[lp_lb_ub_loc + 3] = math.exp(lp_ub[lp_lb_ub_loc + 2])

            # t4=1+t3
            lp_Aeq_t = np.zeros((1, all_variable_num))
            lp_Aeq_t[0][controller_process_location + 4] = 1
            lp_Aeq_t[0][controller_process_location + 3] = -1
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = 1
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
            lp_lb[lp_lb_ub_loc + 4] = 1 + lp_lb[lp_lb_ub_loc + 3]
            lp_ub[lp_lb_ub_loc + 4] = 1 + lp_ub[lp_lb_ub_loc + 3]

            # t5=log(t4)
            xt = all_vars[controller_process_location + 4]
            yt = all_vars[controller_process_location + 5]
            m.addGenConstrLog(xvar=xt, yvar=yt)
            lp_lb[lp_lb_ub_loc + 5] = math.log(lp_lb[lp_lb_ub_loc + 4])
            lp_ub[lp_lb_ub_loc + 5] = math.log(lp_ub[lp_lb_ub_loc + 4])

            # t6=-t5
            lp_Aeq_t = np.zeros((1, all_variable_num))
            lp_Aeq_t[0][controller_process_location + 5] = 1
            lp_Aeq_t[0][controller_process_location + 6] = 1
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = 0
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
            lp_lb[lp_lb_ub_loc + 6] = -lp_ub[lp_lb_ub_loc + 5]
            lp_ub[lp_lb_ub_loc + 6] = -lp_lb[lp_lb_ub_loc + 5]

            # t7=exp(t6)
            xt = all_vars[controller_process_location + 6]
            yt = all_vars[controller_process_location + 7]
            m.addGenConstrExp(xvar=xt, yvar=yt)
            lp_lb[lp_lb_ub_loc + 7] = math.exp(lp_lb[lp_lb_ub_loc + 6])
            lp_ub[lp_lb_ub_loc + 7] = math.exp(lp_ub[lp_lb_ub_loc + 6])

            y_head_record[j] = controller_process_location + 7
            controller_process_location = controller_process_location + 7
            lp_lb_ub_loc = lp_lb_ub_loc + 7

    layer_index = controller_weight_size - 1
    layer_output_before_relu = layer_outputs[layer_index]
    y_head_record_last_layer = y_head_record
    t_process_loc = controller_process_location

    for j in range(0, len(controller_w[layer_index][0])):
        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(controller_w[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = controller_w[layer_index][k, j]

        lp_Aeq_t[0][t_process_loc + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -controller_b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
        lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
        lp_lb_ub_loc = lp_lb_ub_loc + 1
        t_process_loc = t_process_loc + 1
    controller_process_location = controller_process_location + 1




    m.addMConstrs(A=lp_A_Aeq[0:lp_A_b_t_row], x=all_vars, sense='<', b=lp_b_beq[0:lp_A_b_t_row])
    m.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row:], x=all_vars, sense='=', b=lp_b_beq[lp_A_b_t_row:])

    if len(barrier_w) - 1 == 2:
        # this is special for two-layer barrier, no matter how many layers controller
        lp_q1 = np.zeros((all_variable_num, all_variable_num))
        lp_q2 = np.zeros((all_variable_num, all_variable_num))
        k = np.array(barrier_w[2]).reshape([1, -1])
        for i in range(0, len(barrier_w[0][0])):
            for j in range(0, len(barrier_w[1][0])):
                lp_q1[binary[i]][binary[len(barrier_w[0][0]) + j]] = barrier_w[0][0, i] * barrier_w[1][i, j] * k[0][j]
        for i in range(0, len(barrier_w[0][0])):
            for j in range(0, len(barrier_w[1][0])):
                lp_q2[binary[i]][binary[len(barrier_w[0][0]) + j]] = barrier_w[0][1, i] * barrier_w[1][i, j] * k[0][j]

        c_t = np.zeros((1, all_variable_num)).squeeze()
        c_t[process_location + 1] = -1
        m.addMQConstr(Q=lp_q1, c=c_t, sense='=', rhs=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars)
        c_t = np.zeros((1, all_variable_num)).squeeze()
        c_t[process_location + 2] = -1
        m.addMQConstr(Q=lp_q2, c=c_t, sense='=', rhs=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars)
    else:
        # this is special for one-layer barrier, no matter how many layers controller
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

    xt = all_vars[1]
    yt = all_vars[process_location + 3]
    m.addGenConstrSin(xvar=xt, yvar=yt)
    lp_lb[process_location + 3] = -number_close_inf
    lp_ub[process_location + 3] = number_close_inf

    for i in range(0, all_variable_num):
        tvar = all_vars[i]
        tvar.setAttr("lb", lp_lb[i])
        tvar.setAttr("ub", lp_ub[i])

    lp_f = np.zeros((1, all_variable_num)).squeeze()
    q_new = np.zeros((all_variable_num, all_variable_num))
    q_new[process_location + 1, process_location + 3] = 1
    q_new[process_location + 2, process_location + 3 + 7 * controller_hidden_variable_num + 1] = -1
    m.setMObjective(Q=q_new, c=lp_f, constant=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars, sense=GRB.MAXIMIZE)

    m.setParam('FuncPieces', -1)
    m.setParam("FuncPieceError", 0.01)

    m.optimize(mycallback)
    c1 = m.objVal
    c1_gap = m.MIPGap
    c1_v = m.getVars()
    c1_x = []
    for i in range(0, system_dim):
        c1_x.append(c1_v[i].x)

    return c1, c1_gap, c1_x








