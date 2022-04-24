import grid_generate as GridGen
import random
import choose_granularity


class UniformGrid:
    def __init__(self, granularity=None, x_left=None, x_right=None, y_left=None, y_right=None):
        self.granularity = granularity
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right
        self.cell_list = []
        self.cell_num = None

    def construct_grid(self):
        x_unit_len = (self.x_right - self.x_left) / self.granularity
        y_unit_len = (self.y_right - self.y_left) / self.granularity
        total_cell_number = self.granularity * self.granularity
        self.cell_num = total_cell_number
        for i in range(total_cell_number):
            tmp_cell = GridGen.Cell(cell_index=i)
            x_id = i % self.granularity
            y_id = i // self.granularity
            tmp_cell.x_left = self.x_left + x_unit_len * x_id
            tmp_cell.x_right = tmp_cell.x_left + x_unit_len
            tmp_cell.y_left = self.y_left + y_unit_len * y_id
            tmp_cell.y_right = tmp_cell.y_left + y_unit_len
            self.cell_list.append(tmp_cell)

    def generate_x_y_from_cell_index(self, cell_index=None):
        tmp_cell = self.cell_list[cell_index]
        generated_x = random.uniform(tmp_cell.x_left, tmp_cell.x_right)
        generated_y = random.uniform(tmp_cell.y_left, tmp_cell.y_right)
        generated_x = format(generated_x, '.8f')
        generated_y = format(generated_y, '.8f')
        return generated_x, generated_y

    def x_y_2_cell_index(self, x=None, y=None):
        x_unit_len = (self.x_right - self.x_left) / self.granularity
        y_unit_len = (self.y_right - self.y_left) / self.granularity
        tmp_x_id = min(int((x - self.x_left) // x_unit_len), self.granularity - 1)
        tmp_y_id = min(int((y - self.y_left) // y_unit_len), self.granularity - 1)
        tmp_cell_index = tmp_y_id * self.granularity + tmp_x_id
        return tmp_cell_index


class AdaptiveGrid:
    def __init__(self, args=None, n1=None, n2=None):
        self.alpha = args.alpha
        self.x_left = args.x_left
        self.x_right = args.x_right
        self.y_left = args.y_left
        self.y_right = args.y_right
        self.ep = args.epsilon
        self.t = args.trajectory_len
        self.n1 = n1
        self.n2 = n2
        self.level_1_cell_list = []
        self.level_2_cell_list = []
        self.level_1_grid = None
        self.level_2_grid = None

    def construct_level_1_grid(self):
        choose_gran = choose_granularity.choose_granularity_alpha()
        m1 = choose_gran.get_m_1(ep=self.ep, n1=self.n1, alpha_1=self.alpha)
        self.level_1_grid = UniformGrid(granularity=m1, x_left=self.x_left, x_right=self.x_right,
                                        y_left=self.y_left, y_right=self.y_right)
        self.level_1_grid.construct_grid()

    def construct_level_2_grid(self):
        level_1_cell_num = self.level_1_grid.cell_num
        choose_gran = choose_granularity.choose_granularity_alpha()
        tmp_level_2_cell_index = 0
        for k in range(level_1_cell_num):
            tmp_cell = self.level_1_grid.cell_list[k]
            f_k = tmp_cell.noisy_frequency
            m2_k = choose_gran.get_m_2(ep=self.ep, n2=self.n2, f_k=f_k, alpha_2=self.alpha)
            tmp_cell.next_level_grid = UniformGrid(granularity=m2_k, x_left=tmp_cell.x_left, x_right=tmp_cell.x_right,
                                                   y_left=tmp_cell.y_left, y_right=tmp_cell.y_right)
            tmp_cell.next_level_grid.construct_grid()
            for tmp_level_2_cell in tmp_cell.next_level_grid.cell_list:
                tmp_level_2_cell.cell_level_2_index = tmp_level_2_cell_index
                self.level_2_cell_list.append(tmp_level_2_cell)
                tmp_level_2_cell_index += 1
            self.level_1_cell_list.append(tmp_cell)

    def x_y_2_level_1_cell_index(self, x=None, y=None):
        tmp_level_1_grid = self.level_1_grid
        x_unit_len = (tmp_level_1_grid.x_right - tmp_level_1_grid.x_left) / tmp_level_1_grid.granularity
        y_unit_len = (tmp_level_1_grid.y_right - tmp_level_1_grid.y_left) / tmp_level_1_grid.granularity
        tmp_x_id = min(int((x - tmp_level_1_grid.x_left) // x_unit_len), tmp_level_1_grid.granularity - 1)
        tmp_y_id = min(int((y - tmp_level_1_grid.y_left) // y_unit_len), tmp_level_1_grid.granularity - 1)
        tmp_level_1_cell_index = tmp_y_id * tmp_level_1_grid.granularity + tmp_x_id
        return tmp_level_1_cell_index

    def generate_x_y_from_level_2_cell_index(self, level_2_cell_index=None):
        tmp_cell = self.level_2_cell_list[level_2_cell_index]
        generated_x = random.uniform(tmp_cell.x_left, tmp_cell.x_right)
        generated_y = random.uniform(tmp_cell.y_left, tmp_cell.y_right)
        generated_x = format(generated_x, '.8f')
        generated_y = format(generated_y, '.8f')
        return generated_x, generated_y

    def x_y_2_level_2_cell_index(self, x=None, y=None):
        tmp_level_1_grid = self.level_1_grid
        x_unit_len = (tmp_level_1_grid.x_right - tmp_level_1_grid.x_left) / tmp_level_1_grid.granularity
        y_unit_len = (tmp_level_1_grid.y_right - tmp_level_1_grid.y_left) / tmp_level_1_grid.granularity
        tmp_x_id = min(int((x - tmp_level_1_grid.x_left) // x_unit_len), tmp_level_1_grid.granularity - 1)
        tmp_y_id = min(int((y - tmp_level_1_grid.y_left) // y_unit_len), tmp_level_1_grid.granularity - 1)
        tmp_level_1_cell_index = tmp_y_id * tmp_level_1_grid.granularity + tmp_x_id
        tmp_level_2_grid = self.level_1_cell_list[tmp_level_1_cell_index].next_level_grid
        x_unit_len = (tmp_level_2_grid.x_right - tmp_level_2_grid.x_left) / tmp_level_2_grid.granularity
        y_unit_len = (tmp_level_2_grid.y_right - tmp_level_2_grid.y_left) / tmp_level_2_grid.granularity
        tmp_x_id = min(int((x - tmp_level_2_grid.x_left) // x_unit_len), tmp_level_2_grid.granularity - 1)
        tmp_y_id = min(int((y - tmp_level_2_grid.y_left) // y_unit_len), tmp_level_2_grid.granularity - 1)
        tmp_cell_index_in_level_2_grid = tmp_y_id * tmp_level_2_grid.granularity + tmp_x_id
        tmp_level_2_cell_index = tmp_level_2_grid.cell_list[tmp_cell_index_in_level_2_grid].cell_level_2_index
        return tmp_level_2_cell_index
