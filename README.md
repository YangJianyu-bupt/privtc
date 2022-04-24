# PrivTC
[Collecting Individual Trajectories under Local Differential Privacy (Technical Report).pdf](https://github.com/YangJianyu-bupt/privtc/blob/main/PrivTC_technical_report.pdf "悬停显示")


# Guide of Using Code 

**The code is implemented using Python3.7. The descriptions of files are in the following:**


algorithm_PrivGR_PrivSL.py: PrivTC algorithm

PrivGR.py: PrivGR algorithm

PrivSL.py: PrivSL algorithm

choose_granularity.py: set the granularities by guideline

consistency_method.py: norm_sub

frequecy_oracle.py: LDP categorical frequency oracles including OUE and OLH

grid_generate.py: define the cell in the grid

parameter_setting.py: define the papameters

utility_metric_query_avae.py: calculate Query MAE

utility_metric_query_FP.py: calculate FP Similarity

utility_metric_query_length_error.py: calculate Distance Error


The **"example_main.py"** is used to illustrate how to run PrivTC algorithm.
