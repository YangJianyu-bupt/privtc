import numpy as np
import math
import PrivGR


class QueryRegion:
    def __init__(self, x_left=None, x_right=None, y_left=None, y_right=None):
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right


class QueryAvAE:
    def __init__(self, args=None):
        self.args = args
        self.user_num = args.user_num
        self.trajectory_len = args.trajectory_len
        self.x_left = args.x_left
        self.x_right = args.x_right
        self.y_left = args.y_left
        self.y_right = args.y_right
        self.query_region_num = args.query_region_num
        self.query_region_list = []
        self.b = 0.01 * self.args.user_num
        self.random_seed = 1
        self.Grid = None
        self.real_ans_list = []

    def generated_query_region_list(self):
        m1 = math.floor(math.sqrt(self.query_region_num))
        self.Grid = PrivGR.UniformGrid(granularity=m1, x_left=self.x_left, x_right=self.x_right,
                                       y_left=self.y_left, y_right=self.y_right)
        self.Grid.construct_grid()
        self.query_region_list = self.Grid.cell_list

    def get_ans_list_for_each_query_region(self, user_record=None):
        ans_list = np.zeros(self.query_region_num, dtype=int)
        for i in range(self.user_num):
            for j in range(self.trajectory_len):
                loc1 = user_record[i][j].split(',')
                l1 = np.zeros(2, dtype=float)
                l1[0] = float(loc1[0])
                l1[1] = float(loc1[1])
                tmp_cell_id = self.Grid.x_y_2_cell_index(x=l1[0], y=l1[1])
                ans_list[tmp_cell_id] += 1
        ans_list = np.array(ans_list)
        return ans_list

    def get_answer_of_metric(self, ans_real_list=None, ans_syn_list=None):
        AE_list = []
        for i in range(self.query_region_num):
            tmp_AE = np.abs(ans_real_list[i] - ans_syn_list[i]) / self.user_num / self.trajectory_len
            AE_list.append(tmp_AE)
        AE_list = np.array(AE_list)
        AE = np.mean(AE_list)
        return AE
