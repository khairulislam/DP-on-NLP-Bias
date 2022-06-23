import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
# this adds the src folder in the sys path, where the metric_utils.py file is
# not needed if this notebook is in the same folder, but uncomment to access from the data subfolders
sys.path.append( '..' )
from metric_utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

dataset_name = 'ucberkeley'
model_name = 'bert-base-uncased'
raw_id_column = 'comment_id'
dataset_directory = f'../../results/{dataset_name}/'
dataset_directory = '../../../experiment/ucberkeley 1e-3/'
epsilon_list = [0.5, 1.0, 3.0, 6.0, 9.0]

group_map = {
    'gender':['target_gender_men', 'target_gender_women', 'target_gender_transgender'],
    'race':['target_race_white','target_race_black', 'target_race_asian']
}

attributes = list(group_map.keys())
protected_subgroups = []
for attribute in group_map.keys():
    subgroups = group_map[attribute]
    protected_subgroups.extend(subgroups)

print(f'Sensitive attributes {attributes}')
print(f'Protected subgroups {protected_subgroups}')

counts = []
for run in range(1, 4):
    run_folder = f'{dataset_directory}/run {run}'
    model_folder = run_folder # os.path.join(run_folder, model_name)
    normal_folder = os.path.join(model_folder, 'normal')
    result_filepath = os.path.join(normal_folder, 'results.csv')

    result = pd.read_csv(result_filepath)
    result = result[result['split']=='test']
    # drop split column
    result.drop(columns=['split'], inplace=True)

    test_csv_filepath = os.path.join(run_folder, 'test.csv')
    test_df = pd.read_csv(test_csv_filepath)

    test_df.fillna(0, inplace=True)
    # result has id column which is the same as the text ids from raw dataset
    
    test_df.rename({raw_id_column: id_column}, axis=1, inplace=True)
    # if test df has any common columns except id, drop that during merge
    extra_columns = [col for col in test_df.columns if col in result.columns and col!=id_column]

    result = result.merge(test_df.drop(columns=extra_columns), on=id_column, how='inner').reset_index(drop=True)
    result[prediction_column] = result[probability_column] >=0.5
    result = binarize(result, [target_column] + protected_subgroups)

    bias_results = get_all_bias(result, protected_subgroups)
    bias_results.round(3).to_csv(os.path.join(normal_folder, 'bias.csv'), index=False)

    overall_results = get_overall_results(result, protected_subgroups)
    overall_results.round(3).to_csv(os.path.join(normal_folder, 'overall_results.csv'), index=False)

    for epsilon in epsilon_list:
        dp_folder = os.path.join(model_folder, f'epsilon {epsilon}')
        dp_result_filepath = os.path.join(dp_folder, 'results.csv')
        dp_result = pd.read_csv(dp_result_filepath)

        # only calculate test result
        dp_result = dp_result[dp_result['split']=='test']
        dp_result.drop(columns=['split'], inplace=True)
        dp_result = dp_result.merge(test_df.drop(columns=extra_columns), on=id_column, how='inner').reset_index(drop=True)
        
        dp_result[prediction_column] = dp_result[probability_column] >=0.5
        dp_result = binarize(dp_result, [target_column] + protected_subgroups)

        bias_results = get_all_bias(dp_result, protected_subgroups)
        bias_results.round(3).to_csv(os.path.join(dp_folder, 'bias.csv'), index=False)

        overall_results = get_overall_results(dp_result, protected_subgroups)
        overall_results.round(3).to_csv(os.path.join(dp_folder, 'overall_results.csv'), index=False)

    # calculate identity count for all classes
#     train_df = pd.read_csv(os.path.join(run_folder, 'train.csv'))
#     count_df = get_identity_count(train_df, test_df, protected_subgroups)
#     count_df.to_csv(os.path.join(run_folder, 'count.csv'), index=False)
#     counts.append(count_df)


# count_df = pd.concat(counts).groupby('Identity').agg('mean').round().reset_index()
# count_df.to_csv(os.path.join(dataset_directory, 'count.csv'), index=False)