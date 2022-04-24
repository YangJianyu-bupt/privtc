import argparse


def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_num", type=int, default=50000, help="")
    parser.add_argument("--trajectory_len", type=int, default=10, help="")
    parser.add_argument("--consistency_iteration_num_max", type=int, default=2, help="")
    parser.add_argument("--query_region_num", type=int, default=50, help="")
    parser.add_argument("--FP_granularity", type=list, default=[], help="")
    parser.add_argument("--length_bin_num", type=list, default=[], help="")
    parser.add_argument("--epsilon", type=float, default=1.0, help="")
    parser.add_argument("--x_left", type=float, default=0, help="")
    parser.add_argument("--x_right", type=float, default=1.0, help="")
    parser.add_argument("--y_left", type=float, default=0, help="")
    parser.add_argument("--y_right", type=float, default=1.0, help="")
    parser.add_argument("--sigma", type=float, default=0.2, help="")
    parser.add_argument("--alpha", type=float, default=0.02, help="")
    args = parser.parse_args()
    return args
