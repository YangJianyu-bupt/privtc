import numpy as np
import random
import parameter_setting as para
import algorithm_PrivGR_PrivSL
import utility_metric_query_avae as UMquery_avae
import utility_metric_query_FP as UM_FP
import utility_metric_query_length_error as UMlength_error


def setup_args(args=None):
    args.user_num = 5000  # the total number of users
    args.trajectory_len = 6  # the trajectory length
    args.x_left = -45.0  # set the scope of the 2-D geospatial domain
    args.x_right = 85.0
    args.y_left = -160.0
    args.y_right = 160.0

    args.sigma = 0.2  # the parameters in guideline
    args.alpha = 0.02

    args.epsilon = 1.0  # set the privacy budget

    args.query_region_num = 225  # 15 * 15 grid for Query MAE
    args.FP_granularity = 15  # 15 * 15 grid for FP Similarity
    args.length_bin_num = 20  # 20 bins for Distance Error


def load_dataset(txt_dataset_path=None):
    user_record = []
    with open(txt_dataset_path, "r") as fr:
        for line in fr:
            line = line.strip()
            line = line.split()
            user_record.append(line)
    return user_record


def sys_test():
    txt_dataset_path = "txt_dataset/test_dataset_users_5000_tralen_6.txt"
    args = para.generate_args()  # define the parameters
    setup_args(args=args)  # set the parameters
    user_record = load_dataset(txt_dataset_path=txt_dataset_path)  # read user data
    random_seed = 1
    random.seed(random_seed)
    np.random.seed(seed=random_seed)
    np.random.shuffle(user_record)

    # generate query *********************************************
    # Query MAE
    query_mae = UMquery_avae.QueryAvAE(args=args)
    query_mae.generated_query_region_list()
    query_mae.real_ans_list = query_mae.get_ans_list_for_each_query_region(user_record=user_record)

    # Distance Error
    distance_error = UMlength_error.LengthError(args=args)
    distance_error.real_length_list = distance_error.get_length_list(user_record=user_record)

    # FP similarity
    fp_similarity = UM_FP.FP(args=args)
    fp_similarity.construct_grid()
    fp_similarity.real_pattern_sorted_dict = fp_similarity.get_pattern_sorted_dict_with_uniform_grid(user_record=user_record)

    # invoke PrivTC ****************************************************************
    print("PrivTC starts...")
    algo = algorithm_PrivGR_PrivSL.AG_PrivGR_PrivSL(args=args)
    algo.divide_user_record(user_record)
    algo.construct_grid()
    algo.spectral_learning()
    algo.synthesize_data()
    algorithm_user_record = algo.generated_float_user_record_with_level_2_grid
    print("PrivTC ends!")

    # calculate the results of utility metrics **************************************
    print("\nThe results are: ")
    # Query MAE
    algorithm_ans_list = query_mae.get_ans_list_for_each_query_region(user_record=algorithm_user_record)
    mae_ans = query_mae.get_answer_of_metric(ans_real_list=query_mae.real_ans_list, ans_syn_list=algorithm_ans_list)
    print("Query MAE:", mae_ans)

    # FP_similarity
    syn_pattern_sorted_dict = fp_similarity.get_pattern_sorted_dict_with_uniform_grid(user_record=algorithm_user_record)
    fp_ans = fp_similarity.get_FP_similarity(syn_pattern_sorted_dict)
    print("FP similarity:", fp_ans)

    # Distance Error
    syn_length_list = distance_error.get_length_list(user_record=algorithm_user_record)
    dis_ans = distance_error.get_length_error(syn_length_list)
    print("Distance Error:", dis_ans)


if __name__ == '__main__':
    sys_test()
