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
text_column = 'text'
raw_id_column = 'comment_id'

group_map = {
    'gender': {
        'unprivileged':['target_gender_women'],
        'privileged':['target_gender_men']
    },
    'race': {
        'unprivileged':['target_race_black'],
        'privileged': ['target_race_white']
    }
}

identities = []
for group_key in group_map.keys():
    subgroup_map = group_map[group_key]
    for subgroup_key in subgroup_map.keys():
        identities.extend(subgroup_map[subgroup_key])

print(f'Target identities {identities}')

for run in range(1, 4):
    run_folder = f'../../results/{dataset_name}/run {run}'
    model_folder = os.path.join(run_folder, model_name)
    normal_folder = os.path.join(model_folder, 'normal')
    result_filepath = os.path.join(normal_folder, 'results.csv')

    test_csv_filepath = os.path.join(run_folder, 'test.csv')
    test_df = pd.read_csv(test_csv_filepath)
    # Doing this reduces the test file size, makes things easier with git
    # if text_column in test_df.columns:
    #     test_df.drop(columns=text_column, inplace=True)
    #     test_df.to_csv(test_csv_filepath, index=False)

    test_df.fillna(0, inplace=True)
    # result has id column which is the same as the text ids from raw dataset
    
    test_df.rename({raw_id_column: id_column}, axis=1, inplace=True)

    result = pd.read_csv(result_filepath)
    result = result[result['split']=='test']
    # drop split column
    result.drop(columns=['split'], inplace=True)

    # if test df has any common columns except id, drop that during merge
    extra_columns = [col for col in test_df.columns if col in result.columns and col!=id_column]

    result = result.merge(test_df.drop(columns=extra_columns), on=id_column, how='inner').reset_index(drop=True)
    result[prediction_column] = result[probability_column] >=0.5
    result = binarize(result, [target_column] + identities)

    for epsilon in [3.0, 6.0, 9.0]:
        dp_folder = os.path.join(model_folder, f'epsilon {epsilon}')
        dp_result_filepath = os.path.join(dp_folder, 'results.csv')
        dp_result = pd.read_csv(dp_result_filepath)

        # only calculate test result
        dp_result = dp_result[dp_result['split']=='test']
        dp_result.drop(columns=['split'], inplace=True)
        dp_result = dp_result.merge(test_df.drop(columns=extra_columns), on=id_column, how='inner').reset_index(drop=True)
        
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

        overall_results = get_overall_results(group_map, result, dp_result)
        overall_results.round(3).to_csv(os.path.join(dp_folder, 'overall_results.csv'), index=False)

    # calculate identity count for all classes
    train_df = pd.read_csv(os.path.join(run_folder, 'train.csv'))
    # Doing this reduces the file size, makes things easier with git
    # if text_column in train_df.columns:
    #     train_df.drop(columns=text_column, inplace=True)
    #     train_df.to_csv(os.path.join(run_folder, 'train.csv'), index=False)

    count_df = get_identity_count(train_df, test_df, identities)
    count_df.to_csv(os.path.join(run_folder, 'count.csv'), index=False)

    # val_df = pd.read_csv(os.path.join(run_folder, 'validation.csv'))
    # Doing this reduces the file size, makes things easier with git
    # if text_column in val_df.columns:
    #     val_df.drop(columns=text_column).to_csv(os.path.join(run_folder, 'validation.csv'), index=False)