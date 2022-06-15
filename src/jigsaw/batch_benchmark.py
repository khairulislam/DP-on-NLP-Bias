import os, sys
import pandas as pd

sys.path.append('..')
from metric_utils import *

group_map = {
    'gender': {
        'unprivileged':['female'],
        'privileged':['male']
    },
    'race': {
        'unprivileged':['black'],
        'privileged': ['white']
    }
}

identities = []
for group_key in group_map.keys():
    subgroup_map = group_map[group_key]
    for subgroup_key in subgroup_map.keys():
        identities.extend(subgroup_map[subgroup_key])

dataset_name = 'jigsaw unintended bias'
model_name = 'bert'
result_folder = f'../../results/{dataset_name}'
test_csv_filepath = os.path.join(result_folder, 'test.csv')
df = pd.read_csv(test_csv_filepath)
df.fillna(0, inplace=True)

model_folder = os.path.join(result_folder, model_name) # for this particular model
normal_folder = os.path.join(model_folder, 'normal')
result_filepath = os.path.join(normal_folder, 'results.csv')

result = pd.read_csv(result_filepath)
result = result[result['split']=='test']
# drop split column
result.drop(columns=['split'], inplace=True)

result = result.merge(df, on=id_column, how='inner').reset_index(drop=True)
result[prediction_column] = result[probability_column] >=0.5
result = binarize(result, [target_column] + identities)


for epsilon in [3.0, 6.0, 9.0]:
    dp_folder = os.path.join(model_folder, f'epsilon {epsilon}')
    dp_result_filepath = os.path.join(dp_folder, 'results.csv')
    dp_result = pd.read_csv(dp_result_filepath)

    # only calculate test result
    dp_result = dp_result[dp_result['split']=='test']
    dp_result.drop(columns=['split'], inplace=True)
    dp_result = dp_result.merge(df, on=id_column, how='inner').reset_index(drop=True)
    
    dp_result[prediction_column] = dp_result[probability_column] >=0.5
    dp_result = binarize(dp_result, [target_column] + identities)

    bias_results = {
    'fairness_metrics': ['demographic parity', 'Equality of Opportunity (w.r.t y = 1)',
    'Equality of Opportunity (w.r.t y = 0)', 'Equality of Odds', 'unprotected-accuracy',
    'protected-accuracy', 'accuracy']
    }

    for group_key in group_map.keys():
        subgroup_map = group_map[group_key]
        privileged_group = subgroup_map['privileged']
        unprivileged_group = subgroup_map['unprivileged']

        bias_results[group_key] = calculate_bias(result, privileged_group, unprivileged_group)
        bias_results[group_key+'_DP'] = calculate_bias(dp_result, privileged_group, unprivileged_group)

    bias_results = pd.DataFrame(bias_results) 
    bias_results.round(3).to_csv(os.path.join(dp_folder, 'bias.csv'), index=False)


    overall_results = {
        'metrics': ['auc', 'accuracy', 'f1_score', 'precision', 'recall', 'false positive rate']
    }

    for group_key in group_map.keys():
        subgroup_map = group_map[group_key]
        privileged_group = subgroup_map['privileged']
        unprivileged_group = subgroup_map['unprivileged']

        privileged_group_name = ','.join(privileged_group)
        unprivileged_group_name = ','.join(unprivileged_group)

        overall_results[privileged_group_name] = calculate_metrics(result, privileged_group)
        overall_results[privileged_group_name + '_DP'] = calculate_metrics(dp_result, privileged_group)

        overall_results[unprivileged_group_name] = calculate_metrics(result, unprivileged_group)
        overall_results[unprivileged_group_name + '_DP'] = calculate_metrics(dp_result, unprivileged_group)

    overall_results['Total'] = calculate_metrics(result, [])
    overall_results['Total_DP'] = calculate_metrics(dp_result, [])

    overall_results = pd.DataFrame(overall_results) 
    overall_results.columns = [col.replace('target_', '') for col in overall_results.columns]
    overall_results.round(3).to_csv(os.path.join(dp_folder, 'overall_results.csv'), index=False)