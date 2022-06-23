import os, sys
import pandas as pd

sys.path.append('..')
from metric_utils import *

group_map = {
    'gender':['male', 'female', 'transgender'],
    'race':['white', 'black', 'asian']
}

attributes = list(group_map.keys())
protected_subgroups = []
for attribute in group_map.keys():
    subgroups = group_map[attribute]
    protected_subgroups.extend(subgroups)

print(f'Sensitive attributes {attributes}')
print(f'Protected subgroups {protected_subgroups}')

dataset_name = 'jigsaw'
model_name = 'bert-base-uncased'
epsilon_list = [0.5, 1.0, 3.0, 6.0, 9.0]
runs = 3

test_csv_filepath = os.path.join(f'../../results/{dataset_name}', 'test.csv')
test_df = pd.read_csv(test_csv_filepath)
test_df.fillna(0, inplace=True)

for run in range(1, runs+1):
    run_folder = f'../../results/{dataset_name}/run {run}'
    model_folder = os.path.join(run_folder, model_name)
    normal_folder = os.path.join(model_folder, 'normal')
    result_filepath = os.path.join(normal_folder, 'results.csv')

    result = pd.read_csv(result_filepath)
    result = result[result['split']=='test']
    # drop split column
    result.drop(columns=['split'], inplace=True)

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