import numpy as np
import PrivGR


class QueryRegion:
    def __init__(self, x_left=None, x_right=None, y_left=None, y_right=None):
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right


class FP:
    def __init__(self, args=None):
        self.args = args
        self.user_num = args.user_num
        self.x_left = args.x_left
        self.x_right = args.x_right
        self.y_left = args.y_left
        self.y_right = args.y_right
        self.m1 = args.FP_granularity
        self.P_len = 2  # pattern length
        self.k = 100
        self.real_pattern_sorted_dict = None
        self.syn_pattern_sorted_dict = None

    def construct_grid(self):
        self.Grid = PrivGR.UniformGrid(granularity=self.m1, x_left=self.x_left, x_right=self.x_right,
                                       y_left=self.y_left, y_right=self.y_right)
        self.Grid.construct_grid()

    def get_pattern_sorted_dict_with_uniform_grid(self, user_record=None):
        user_num = self.args.user_num
        trajectory_len = self.args.trajectory_len
        discretized_trajectory_with_uniform_grid = np.zeros(shape=(user_num, trajectory_len), dtype=int)
        for i in range(user_num):
            for j in range(trajectory_len):
                tmp_x_y = user_record[i][j].split(',')
                x = float(tmp_x_y[0])
                y = float(tmp_x_y[1])
                discretized_trajectory_with_uniform_grid[i][j] = self.Grid.x_y_2_cell_index(x, y)
        pattern_support = np.zeros(shape=(self.Grid.cell_num, self.Grid.cell_num), dtype=int)
        for i in range(user_num):
            for j in range(trajectory_len-1):
                tmp_0 = discretized_trajectory_with_uniform_grid[i][j]
                tmp_1 = discretized_trajectory_with_uniform_grid[i][j + 1]
                pattern_support[tmp_0][tmp_1] += 1
        P_dict = dict()
        weight_0 = self.Grid.cell_num
        weight_1 = 1
        for tmp_0 in range(self.Grid.cell_num):
            for tmp_1 in range(self.Grid.cell_num):
                tmp_P_id = tmp_0 * weight_0 + tmp_1 * weight_1
                P_dict[tmp_P_id] = pattern_support[tmp_0][tmp_1]
        pattern_sorted_dict = dict(sorted(P_dict.items(), key=lambda kv: kv[1], reverse=True))
        return pattern_sorted_dict

    def get_pattern_sorted_dict_with_uniform_grid_P_len_3(self, user_record=None):
        user_num = self.args.user_num
        trajectory_len = self.args.trajectory_len
        discretized_trajectory_with_uniform_grid = np.zeros(shape=(user_num, trajectory_len), dtype=int)
        for i in range(user_num):
            for j in range(trajectory_len):
                tmp_x_y = user_record[i][j].split(',')
                x = float(tmp_x_y[0])
                y = float(tmp_x_y[1])
                discretized_trajectory_with_uniform_grid[i][j] = self.Grid.x_y_2_cell_index(x, y)
        pattern_support = np.zeros(shape=(self.Grid.cell_num, self.Grid.cell_num, self.Grid.cell_num), dtype=int)
        for i in range(user_num):
            for j in range(trajectory_len-2):
                tmp_0 = discretized_trajectory_with_uniform_grid[i][j]
                tmp_1 = discretized_trajectory_with_uniform_grid[i][j + 1]
                tmp_2 = discretized_trajectory_with_uniform_grid[i][j + 2]
                pattern_support[tmp_0][tmp_1][tmp_2] += 1
        P_dict = dict()
        weight_0 = self.Grid.cell_num * self.Grid.cell_num
        weight_1 = self.Grid.cell_num
        for tmp_0 in range(self.Grid.cell_num):
            for tmp_1 in range(self.Grid.cell_num):
                for tmp_2 in range(self.Grid.cell_num):
                    tmp_P_id = tmp_0 * weight_0 + tmp_1 * weight_1 + tmp_2
                    P_dict[tmp_P_id] = pattern_support[tmp_0][tmp_1][tmp_2]
        pattern_sorted_dict = dict(sorted(P_dict.items(), key=lambda kv: kv[1], reverse=True))
        return pattern_sorted_dict

    def get_FP_avre(self, syn_pattern_dict: dict = None):
        ans = 0
        real_pattern_list = list(self.real_pattern_sorted_dict.keys())
        for i in range(self.k):
            tmp_pattern = real_pattern_list[i]
            tt = np.abs(self.real_pattern_sorted_dict[tmp_pattern] - syn_pattern_dict[tmp_pattern])
            if self.real_pattern_sorted_dict[tmp_pattern] > 0:
                ans += tt / self.real_pattern_sorted_dict[tmp_pattern]
        ans /= self.k
        return ans

    def get_FP_avae(self, syn_pattern_dict: dict = None):
        ans = 0
        real_pattern_list = list(self.real_pattern_sorted_dict.keys())
        for i in range(self.k):
            tmp_pattern = real_pattern_list[i]
            tt = np.abs(self.real_pattern_sorted_dict[tmp_pattern] - syn_pattern_dict[tmp_pattern])
            ans += tt / self.user_num
        ans /= self.k
        return ans

    def get_FP_similarity(self, syn_pattern_dict: dict = None):
        real_pattern_list = list(self.real_pattern_sorted_dict.keys())
        syn_pattern_list = list(syn_pattern_dict.keys())
        real_frequent_pattern_list = real_pattern_list[0:self.k]
        syn_frequent_pattern_list = syn_pattern_list[0:self.k]
        real_not_frequent_pattern_list = real_pattern_list[self.k:]
        syn_not_frequent_patter_list = syn_pattern_list[self.k:]
        TP = len(set(real_frequent_pattern_list) & set(syn_frequent_pattern_list))
        FN = len(set(real_frequent_pattern_list) & set(syn_not_frequent_patter_list))
        FP = len(set(real_not_frequent_pattern_list) & set(syn_frequent_pattern_list))
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        if P + R == 0:
            F1_score = 0
        else:
            F1_score = 2 * P * R / (P + R)
        return F1_score
