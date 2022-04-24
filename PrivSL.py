import numpy as np


def get_distribution(data, observation_number):
    """
    p[0] = Pr[x_1 = i]
    p[1] = Pr[x_2 = i, x_1 = j]
    p[2] = Pr[x_3 = i, x_2 = x, x_1 = j]
    """
    p = [0, 0, 0]
    p[0] = np.zeros(shape=(observation_number))
    p[1] = np.zeros(shape=(observation_number, observation_number))
    p[2] = np.zeros(shape=(observation_number, observation_number, observation_number))
    sample_number = 0
    for seq in data:
        assert len(seq) >= 3
        pos = np.random.randint(0, len(seq) - 2)
        p[0][seq[pos]] += 1
        p[1][seq[pos + 1]][seq[pos]] += 1
        p[2][seq[pos + 1]][seq[pos + 2]][seq[pos]] += 1
        sample_number += 1
    for i in range(3):
        p[i] /= sample_number
    return p


def learnhmm(p, observation_number, hidden_number):
    '''
    Compute the SVD of p[1], and let Ub be the matrix of left singular
    vectors corresponding to the m largest singular values.
    '''
    U, s, a = np.linalg.svd(p[1])
    U = U[:, :hidden_number]
    '''
    b_1 = U^T * p[0]
    b_n = (p[1]^T * U)^-1 * p[0]
    B_x = (U^T * p[2]_x) * (U^T * p[1])^-1
    '''
    b_1 = np.matmul(np.transpose(U), p[0])
    b_n = np.matmul(np.linalg.pinv(np.matmul(np.transpose(p[1]), U)), p[0])
    B = []
    for x in range(observation_number):
        b_x = np.matmul(np.matmul(np.transpose(U), p[2][x]), np.linalg.pinv(np.matmul(np.transpose(U), p[1])))
        B.append(b_x)
    return b_1, b_n, B


def rebuild_data(b_1, b_n, B, user_number, observation_number, sequence_length):
    data = []
    mid_p_x_list = []
    for x in range(observation_number):
        mid_p_x = np.matmul(np.transpose(b_n), B[x])
        mid_p_x_list.append(mid_p_x)
    for i in range(user_number):
        seq = []
        b = b_1
        for _ in range(sequence_length):
            '''
            Pr[x_t|x_1...t-1] is proportional to b_n^T * B[x_t] * b_t
            '''
            p = np.zeros(shape=(observation_number))
            for x in range(observation_number):
                p[x] = np.matmul(mid_p_x_list[x], b)
            p = np.maximum(p, 0)
            t_sum = np.sum(p)
            if t_sum == 0:
                p[:] = 1.0 / observation_number
            else:
                p /= np.sum(p)
            x_t = np.random.choice(observation_number, p=p)
            seq.append(x_t)
            tmp_r = np.matmul(mid_p_x_list[x_t], b)
            if tmp_r == 0:
                pass
            else:
                b = np.matmul(B[x_t], b) / tmp_r
        data.append(seq)
    return data
