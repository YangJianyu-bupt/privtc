import numpy as np
import math


class LengthError:
    def __init__(self, args=None):
        self.args = args
        self.user_num = args.user_num
        self.x_left = args.x_left
        self.x_right = args.x_right
        self.y_left = args.y_left
        self.y_right = args.y_right
        self.bin_num = args.length_bin_num
        self.real_length_list = None

    def get_length_list(self, user_record=None):
        user_num = self.args.user_num
        trajectory_len = self.args.trajectory_len
        length_list = []
        for i in range(user_num):
            tmp_length = 0
            for j in range(trajectory_len - 1):
                tmp_x1_y1 = user_record[i][j].split(',')
                x1 = float(tmp_x1_y1[0])
                y1 = float(tmp_x1_y1[1])
                tmp_x2_y2 = user_record[i][j + 1].split(',')
                x2 = float(tmp_x2_y2[0])
                y2 = float(tmp_x2_y2[1])
                tmp_length += math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
            length_list.append(tmp_length)
        length_list = np.array(length_list)
        return length_list

    def get_length_error(self, syn_length_list=None):
        max_length = self.real_length_list.max()
        real_hist, real_bin_edges = np.histogram(self.real_length_list, range=(0, max_length), bins=self.bin_num, density=False)
        P = real_hist / np.sum(real_hist)
        syn_length_list[syn_length_list > max_length] = max_length
        syn_hist, syn_bin_edges = np.histogram(syn_length_list, range=(0, max_length), bins=self.bin_num, density=False)
        Q = syn_hist / np.sum(syn_hist)

        def KLD(p, q):
            p, q = zip(*filter(lambda tt: tt[0] != 0 or tt[1] != 0, zip(p, q)))
            return sum([_p * math.log(_p / _q, 2) for (_p, _q) in zip(p, q)])

        def JSD(p, q):
            p, q = zip(*filter(lambda tt: tt[0] != 0 or tt[1] != 0, zip(p, q)))
            p = np.array(p)
            q = np.array(q)
            M = (p + q) * 0.5
            p = p + np.spacing(1)
            q = q + np.spacing(1)
            M = M + np.spacing(1)
            tans1 = 0.5 * KLD(p, M)
            tans2 = 0.5 * KLD(q, M)
            return tans1 + tans2
        ans = JSD(P, Q)
        return ans
