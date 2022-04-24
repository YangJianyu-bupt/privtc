class Cell:
    def __init__(self, x_left=None, x_right=None, y_left=None, y_right=None, cell_index=None):
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right
        self.cell_index = cell_index
        self.real_frequency = None
        self.noisy_frequency = None
        self.next_level_grid = None
        self.cell_level_2_index = None
