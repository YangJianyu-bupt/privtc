import numpy as np
import PrivGR
import PrivSL
import copy
import math
import frequency_oracle as FreOra
import consistency_method as ConMeth


class AG_PrivGR_PrivSL:
    def __init__(self, args=None):
        self.args = args
        self.Grid = None
        self.discretized_phase_1_user_record_with_level_1_grid = None
        self.discretized_phase_2_user_record_with_level_2_grid = None
        self.generated_discretized_user_record_with_level_2_grid = None
        self.generated_float_user_record_with_level_2_grid = None
        self.phase_1_user_record = None
        self.phase_2_user_record = None
        self.portion_of_phase_1_user = args.sigma
        self.phase_1_user_num = None
        self.phase_2_user_num = None

        self.marginal_p_x_user_num = None
        self.marginal_p_y_x_user_num = None
        self.marginal_p_y_z_x_user_num = None

        self.marginal_p_x_user_record = None
        self.marginal_p_y_x_way_user_record = None
        self.marginal_p_y_z_x_way_user_record = None

        self.LDP_mechanism_list = None
        self.observation_number = None
        self.hidden_number = 10
        self.p_x = None
        self.p_y_x = None
        self.p_y_z_x = None

    def divide_user_record(self, user_record: list = None):
        self.phase_1_user_num = int(self.portion_of_phase_1_user * self.args.user_num)
        self.phase_2_user_num = self.args.user_num - self.phase_1_user_num

        self.phase_1_user_record = copy.deepcopy(user_record[0: self.phase_1_user_num])
        self.phase_2_user_record = copy.deepcopy(user_record[self.phase_1_user_num:])

    def construct_grid(self):
        # here is why it is called pro
        self.Grid = PrivGR.AdaptiveGrid(args=self.args, n1=self.args.user_num, n2=self.phase_2_user_num)
        self.Grid.construct_level_1_grid()
        self.discretize_trajectory_with_level_1_grid()
        self.get_noisy_frequency_of_cell_in_level_1_grid()
        self.Grid.construct_level_2_grid()
        self.discretize_trajectory_with_level_2_grid()

    def discretize_trajectory_with_level_1_grid(self):
        user_num = self.phase_1_user_num
        trajectory_len = self.args.trajectory_len
        self.discretized_phase_1_user_record_with_level_1_grid = np.zeros(shape=(user_num, trajectory_len), dtype=int)
        for i in range(user_num):
            for j in range(trajectory_len):
                tmp_x_y = self.phase_1_user_record[i][j].split(',')
                x = float(tmp_x_y[0])
                y = float(tmp_x_y[1])
                self.discretized_phase_1_user_record_with_level_1_grid[i][j] = self.Grid.x_y_2_level_1_cell_index(x, y)

    def discretize_trajectory_with_level_2_grid(self):
        user_num = self.phase_2_user_num
        trajectory_len = self.args.trajectory_len
        self.discretized_phase_2_user_record_with_level_2_grid = np.zeros(shape=(user_num, trajectory_len), dtype=int)
        for i in range(user_num):
            for j in range(trajectory_len):
                tmp_x_y = self.phase_2_user_record[i][j].split(',')
                x = float(tmp_x_y[0])
                y = float(tmp_x_y[1])
                self.discretized_phase_2_user_record_with_level_2_grid[i][j] = self.Grid.x_y_2_level_2_cell_index(x, y)

    def get_noisy_frequency_of_cell_in_level_1_grid(self):
        self.LDP_mechanism_list = []  # initialize for each time to randomize user data
        epsilon = self.args.epsilon
        group_num = self.args.trajectory_len
        user_num = self.phase_1_user_num
        tmp_domain_size = self.Grid.level_1_grid.cell_num

        for j in range(group_num):  # initialize LDP mechanism for each group
            # tmp_LDR = FreOra.OUE(domain_size= tmp_domain_size, epsilon= epsilon, user_num= user_num, sampling_factor= group_num)
            tmp_LDR = FreOra.OLH(domain_size=tmp_domain_size, epsilon=epsilon, user_num=user_num, sampling_factor=group_num)
            self.LDP_mechanism_list.append(tmp_LDR)

        for i in range(user_num):
            tmp_user_granularity = math.ceil(user_num / group_num)
            group_index_of_user = i // tmp_user_granularity
            j = group_index_of_user
            self.LDP_mechanism_list[j].group_user_num += 1  # count the user num of each group
            tmp_real_cell_index = self.discretized_phase_1_user_record_with_level_1_grid[i][j]
            tmp_LDP_mechanism = self.LDP_mechanism_list[j]
            tmp_LDP_mechanism.operation_perturb(tmp_real_cell_index)

        # update the perturbed_count of each cell
        average_aggregated_prob = None
        for j in range(group_num):
            tmp_LDP_mechanism = self.LDP_mechanism_list[j]
            tmp_LDP_mechanism.operation_aggregate()
            if j == 0:
                average_aggregated_prob = tmp_LDP_mechanism.aggregated_prob
            else:
                average_aggregated_prob = average_aggregated_prob + tmp_LDP_mechanism.aggregated_prob
        average_aggregated_prob = average_aggregated_prob / group_num
        cons_aggregated_prob = ConMeth.norm_sub_frequency(average_aggregated_prob, tolerance=1.0 / user_num)
        for i in range(tmp_domain_size):
            self.Grid.level_1_grid.cell_list[i].noisy_frequency = cons_aggregated_prob[i]

    def spectral_learning(self):
        self.observation_number = len(self.Grid.level_2_cell_list)
        self.divide_discretized_phase_2_user_record()
        self.p_x = np.zeros(shape=(self.observation_number), dtype=float)
        self.p_y_x = np.zeros(shape=(self.observation_number, self.observation_number), dtype=float)
        self.p_y_z_x = np.zeros(shape=(self.observation_number, self.observation_number, self.observation_number), dtype=float)
        self.get_noisy_p_1_distribution()
        self.get_noisy_p_2_distribution()
        self.get_noisy_p_3_distribution()
        self.get_consistent_p_1_and_p_2_and_p_3()
        p = [0, 0, 0]
        p[0] = self.p_x
        p[1] = self.p_y_x
        p[2] = self.p_y_z_x
        b_1, b_n, B = PrivSL.learnhmm(p, self.observation_number, self.hidden_number)
        self.generated_discretized_user_record_with_level_2_grid = PrivSL.rebuild_data(b_1, b_n, B, self.args.user_num, self.observation_number, self.args.trajectory_len)

    def divide_discretized_phase_2_user_record(self):
        tmp_user_granularity = int(self.phase_2_user_num / 3)
        self.marginal_p_x_user_num = tmp_user_granularity
        self.marginal_p_y_x_user_num = tmp_user_granularity
        self.marginal_p_y_z_x_user_num = self.phase_2_user_num - tmp_user_granularity * 2
        self.marginal_p_x_user_record = copy.deepcopy(self.discretized_phase_2_user_record_with_level_2_grid[0:tmp_user_granularity])
        self.marginal_p_y_x_way_user_record = copy.deepcopy(self.discretized_phase_2_user_record_with_level_2_grid[tmp_user_granularity: tmp_user_granularity * 2])
        self.marginal_p_y_z_x_way_user_record = copy.deepcopy(self.discretized_phase_2_user_record_with_level_2_grid[tmp_user_granularity * 2:])

    def get_noisy_p_x_distribution(self):
        epsilon = self.args.epsilon
        user_num = self.marginal_p_x_user_num
        trajectory_len = self.args.trajectory_len
        tmp_domain_size = self.observation_number
        # tmp_LDP = FreOra.OUE(domain_size=tmp_domain_size, epsilon=epsilon, user_num=user_num)
        tmp_LDP = FreOra.OLH(domain_size=tmp_domain_size, epsilon=epsilon, user_num=user_num)
        for i in range(user_num):
            j = np.random.randint(0, trajectory_len - 2)
            tmp_LDP.group_user_num += 1
            tmp_real_cell_index = self.discretized_phase_2_user_record_with_level_2_grid[i][j]
            tmp_LDP.operation_perturb(tmp_real_cell_index)
        tmp_LDP.operation_aggregate()
        average_aggregated_prob = tmp_LDP.aggregated_prob
        cons_aggregated_prob = ConMeth.norm_sub_frequency(average_aggregated_prob, tolerance=1.0 / user_num)
        for i in range(tmp_domain_size):
            self.p_x[i] = cons_aggregated_prob[i]

    def get_noisy_p_1_distribution(self):
        self.get_noisy_p_x_distribution()

    def get_noisy_p_y_x_distribution(self):
        epsilon = self.args.epsilon
        user_num = self.marginal_p_y_x_user_num
        trajectory_len = self.args.trajectory_len
        tmp_domain_size = self.observation_number * self.observation_number
        # tmp_LDP = FreOra.OUE(domain_size=tmp_domain_size, epsilon=epsilon, user_num=user_num)
        tmp_LDP = FreOra.OLH(domain_size=tmp_domain_size, epsilon=epsilon, user_num=user_num)
        for i in range(user_num):
            j = np.random.randint(0, trajectory_len - 2)
            tmp_LDP.group_user_num += 1
            tmp_y = self.discretized_phase_2_user_record_with_level_2_grid[i][j + 1]
            tmp_x = self.discretized_phase_2_user_record_with_level_2_grid[i][j]
            tmp_real_val_index = tmp_y * self.observation_number + tmp_x
            tmp_LDP.operation_perturb(tmp_real_val_index)
        tmp_LDP.operation_aggregate()
        average_aggregated_prob = tmp_LDP.aggregated_prob
        cons_aggregated_prob = ConMeth.norm_sub_frequency(average_aggregated_prob, tolerance=1.0 / user_num)
        for tmp_y in range(self.observation_number):
            for tmp_x in range(self.observation_number):
                self.p_y_x[tmp_y][tmp_x] = cons_aggregated_prob[tmp_y * self.observation_number + tmp_x]

    def get_noisy_p_2_distribution(self):
        self.get_noisy_p_y_x_distribution()

    def get_noisy_p_y_z_x_distribution(self):
        epsilon = self.args.epsilon
        user_num = self.marginal_p_y_z_x_user_num
        trajectory_len = self.args.trajectory_len
        tmp_domain_size = self.observation_number * self.observation_number * self.observation_number
        # tmp_LDP = FreOra.OUE(domain_size=tmp_domain_size, epsilon=epsilon, user_num=user_num)
        tmp_LDP = FreOra.OLH(domain_size=tmp_domain_size, epsilon=epsilon, user_num=user_num)
        for i in range(user_num):
            j = np.random.randint(0, trajectory_len - 2)
            tmp_LDP.group_user_num += 1
            tmp_y = self.discretized_phase_2_user_record_with_level_2_grid[i][j + 1]
            tmp_z = self.discretized_phase_2_user_record_with_level_2_grid[i][j + 2]
            tmp_x = self.discretized_phase_2_user_record_with_level_2_grid[i][j]
            weight_y = self.observation_number * self.observation_number
            weight_z = self.observation_number
            weight_x = 1
            tmp_real_val_index = tmp_y * weight_y + tmp_z * weight_z + tmp_x * weight_x
            tmp_LDP.operation_perturb(tmp_real_val_index)
        tmp_LDP.operation_aggregate()
        average_aggregated_prob = tmp_LDP.aggregated_prob
        cons_aggregated_prob = ConMeth.norm_sub_frequency(average_aggregated_prob, tolerance=1.0 / self.args.user_num, flag_p3=True)
        for tmp_y in range(self.observation_number):
            for tmp_z in range(self.observation_number):
                for tmp_x in range(self.observation_number):
                    self.p_y_z_x[tmp_y, tmp_z, tmp_x] = cons_aggregated_prob[tmp_y * weight_y + tmp_z * weight_z + tmp_x * weight_x]

    def get_noisy_p_3_distribution(self):
        self.get_noisy_p_y_z_x_distribution()

    def overall_consistency(self):
        for i in range(self.observation_number):  # for x
            s_x = self.observation_number * self.observation_number
            s_y_x = self.observation_number
            s_y_z_x = 1
            weight_x = (s_x / (s_x + s_y_x + s_y_z_x))
            weight_y_x = (s_y_x / (s_x + s_y_x + s_y_z_x))
            weight_y_z_x = (s_y_z_x / (s_x + s_y_x + s_y_z_x))
            ave = self.p_x[i] * weight_x + np.sum(self.p_y_x[:, i]) * weight_y_x + np.sum(self.p_y_z_x[:, :, i]) * weight_y_z_x
            self.p_x[i] = ave
            diff = (ave - np.sum(self.p_y_x[:, i])) / self.observation_number
            self.p_y_x[:, i] = self.p_y_x[:, i] + diff
            diff = (ave - np.sum(self.p_y_z_x[:, :,  i])) / self.observation_number / self.observation_number
            self.p_y_z_x[:, :, i] = self.p_y_z_x[:, :, i] + diff

        for i in range(self.observation_number):  # for y
            s_y_x = self.observation_number
            s_y_z_x = 1
            weight_y_x = (s_y_x / (s_y_x + s_y_z_x))
            weight_y_z_x = (s_y_z_x / (s_y_x + s_y_z_x))
            ave = np.sum(self.p_y_x[i, :]) * weight_y_x + np.sum(self.p_y_z_x[i, :, :]) * weight_y_z_x
            diff = (ave - np.sum(self.p_y_x[i, :])) / self.observation_number
            self.p_y_x[i, :] = self.p_y_x[i, :] + diff
            diff = (ave - np.sum(self.p_y_z_x[i, :, :])) / self.observation_number / self.observation_number
            self.p_y_z_x[i, :, :] = self.p_y_z_x[i, :, :] + diff

    def get_consistent_p_1_and_p_2_and_p_3(self):
        theta = 1.0 / self.args.user_num
        if self.observation_number > 200:
            consistency_iteration_num_max = 1
        else:
            consistency_iteration_num_max = self.args.consistency_iteration_num_max
        for i in range(consistency_iteration_num_max):
            self.overall_consistency()
            tmp_p_x = copy.deepcopy(self.p_x)
            self.p_x = ConMeth.norm_sub_frequency(tmp_p_x, tolerance=theta)
            tmp_p_y_x = copy.deepcopy(self.p_y_x)
            tmp_p_y_x_1_way_array = tmp_p_y_x.reshape((tmp_p_y_x.size))
            new_tmp_p_y_x_1_way_array = ConMeth.norm_sub_frequency(tmp_p_y_x_1_way_array, tolerance=theta)
            self.p_y_x = new_tmp_p_y_x_1_way_array.reshape(tmp_p_y_x.shape)
            tmp_p_y_z_x = copy.deepcopy(self.p_y_z_x)
            tmp_p_y_z_x_1_way_array = tmp_p_y_z_x.reshape((tmp_p_y_z_x.size))
            new_tmp_p_y_z_x_1_way_array = ConMeth.norm_sub_frequency(tmp_p_y_z_x_1_way_array, tolerance=theta, flag_p3=True)
            self.p_y_z_x = new_tmp_p_y_z_x_1_way_array.reshape(tmp_p_y_z_x.shape)

    def synthesize_data(self):
        tmp_list = []
        for i in range(self.args.user_num):
            tmp_record = []
            for j in range(self.args.trajectory_len):
                tmp_val = self.generated_discretized_user_record_with_level_2_grid[i][j]
                tmp_x, tmp_y = self.Grid.generate_x_y_from_level_2_cell_index(level_2_cell_index=tmp_val)
                tmp_str = str(tmp_x) + ',' + str(tmp_y)
                tmp_record.append(tmp_str)
            tmp_list.append(tmp_record)
        self.generated_float_user_record_with_level_2_grid = np.array(tmp_list)
        return
