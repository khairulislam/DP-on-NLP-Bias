from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
from collections import defaultdict

prediction_column = 'preds'
probability_column = 'probs'
target_column = 'labels'
id_column = 'id' # result csv files always has id in this same column. can be different in train or test csv files

def get_overall_results(group_map, result):
    overall_results = {
        'metrics': ['auc', 'accuracy', 'f1_score', 
        'precision', 'recall', 'false positive rate',
        'bnsp_auc', 'bpsn_auc']
    }

    for group_key in group_map.keys():
        subgroup_map = group_map[group_key]
        privileged_group = subgroup_map['privileged']
        unprivileged_group = subgroup_map['unprivileged']

        privileged_group_name = ','.join(privileged_group)
        unprivileged_group_name = ','.join(unprivileged_group)

        overall_results[privileged_group_name] = calculate_metrics(result, privileged_group)
        overall_results[unprivileged_group_name] = calculate_metrics(result, unprivileged_group)

    overall_results['Total'] = calculate_metrics(result, [])

    overall_results = pd.DataFrame(overall_results) 
    # overall_results.columns = [col.replace('target_', '') for col in overall_results.columns]
    return overall_results

# assume we are given label_set_dict, a dict mapping {label: title}
# label_set_dict = {0: 'professor', 1: 'accountant' ...}
def get_identity_count(train_df, test_df, identities, label_set_dict):
    count_dict = defaultdict(list)
    for identity in identities:
        train_counts = train_df[train_df[identity] >= 0.5][target_column].value_counts().to_numpy()
        train_titles = train_df[train_df[identity] >= 0.5][target_column].value_counts().index.to_numpy()
        for i in range(len(train_titles))
            title = train_counts[i]
            count_dict[f'{title} (train)'].append(train_counts[i])
        
        test_counts = test_df[test_df[identity] >= 0.5][test_column].value_counts().to_numpy()
        test_titles = test_df[test_df[identity] >= 0.5][test_column].value_counts().index.to_numpy()
        for i in range(len(test_titles))
            title = test_counts[i]
            count_dict[f'{title} (test)'].append(test_counts[i])    
            
    return pd.DataFrame(count_dict)

def binarize(df, columns):
    for col in columns:
        df[col] = (df[col] >= 0.5)
    return df

# https://en.wikipedia.org/wiki/Confusion_matrix
def false_positive_rate(y_true, y_pred, label_set):
    FPR_list = []
    cm = confusion_matrix(y_true, y_pred, labels=label_set)
    T = cm.sum(axis=1)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FPR = (FP / (FP + TN + 1e-8)) # * T / np.sum(T) (weighted)
    return np.mean(FPR)

def true_positive_rate(y_true, y_pred, label_set):
    FPR_list = []
    cm = confusion_matrix(y_true, y_pred, labels=label_set)
    T = cm.sum(axis=1)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    TPR = (TP / (TP + FN + 1e-8)) # * T / np.sum(T) (weighted)
    return np.mean(FPR)

# title is an integer that corresponds to the label
def calculate_demographic_parity(df_privileged, df_unprivileged, title):
    privileged_number = df_privileged.shape[0]
    if privileged_number >0:
        privileged_positive_percent = sum(df_privileged[prediction_column] == title) / privileged_number
    else:
        print('Warning, no privileged example found')
        privileged_positive_percent = 0

    unprivileged_number = df_privileged.shape[0]
    if unprivileged_number >0:
        unprivileged_positive_percent = sum(df_unprivileged[prediction_column] == title) / unprivileged_number
    else:
        print('Warning, no unprivileged example found')
        unprivileged_positive_percent = 0
    
    demographic_parity = 1 - abs(privileged_positive_percent-unprivileged_positive_percent)
    return demographic_parity

def calculate_equality(df_privileged, df_unprivileged):
    fpr_privileged = false_positive_rate(df_privileged[target_column], df_privileged[prediction_column])
    tpr_privileged = true_positive_rate(df_privileged[target_column], df_privileged[prediction_column])

    fpr_unprivileged = false_positive_rate(df_unprivileged[target_column], df_unprivileged[prediction_column])
    tpr_unprivileged = true_positive_rate(df_unprivileged[target_column], df_unprivileged[prediction_column])

    EqOpp0 = 1 - abs(fpr_privileged - fpr_unprivileged)
    EqOpp1 = 1 - abs(tpr_privileged - tpr_unprivileged)
    EqOdd = (EqOpp0 + EqOpp1) / 2

    return EqOpp1, EqOpp0, EqOdd

def calculate_sensitive_accuracy(df_privileged, df_unprivileged):
    accuracy_privileged = accuracy_score(df_privileged[target_column], df_privileged[prediction_column])
    accuracy_unprivileged = accuracy_score(df_unprivileged[target_column], df_unprivileged[prediction_column])

    accuracy = (accuracy_privileged + accuracy_unprivileged) / 2
    return accuracy_unprivileged, accuracy_privileged, accuracy

def get_all_bias(group_map, df):
    bias_results = {
        'fairness_metrics': ['DP', 'EqOpp1',
        'EqOpp0', 'EqOdd', 'up-accuracy',
        'p-accuracy', 'accuracy']
        }

    for group_key in group_map.keys():
        subgroup_map = group_map[group_key]
        privileged_group = subgroup_map['privileged']
        unprivileged_group = subgroup_map['unprivileged']

        bias_results[group_key] = calculate_bias(df, privileged_group, unprivileged_group)

    return pd.DataFrame(bias_results) 

def calculate_bias(df, privileged_group, unprivileged_group):
    df_privileged = df[(df[privileged_group]).any(axis=1)]
    df_unprivileged = df[(df[unprivileged_group]).any(axis=1)]

    demographic_parity = calculate_demographic_parity(df_privileged, df_unprivileged)
    EqOpp1, EqOpp0, EqOdd = calculate_equality(df_privileged, df_unprivileged)
    accuracy_unprivileged, accuracy_privileged, accuracy = calculate_sensitive_accuracy(df_privileged, df_unprivileged)

    biases = [demographic_parity, EqOpp1, EqOpp0, EqOdd, accuracy_unprivileged, accuracy_privileged, accuracy]
    return biases

# SUBGROUP_AUC = 'auc'
# BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
# BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

# def calculate_metrics(total_df, group):
#     # if group is empty return the overall metric. 
#     # otherwise only calculate for example of that group
#     if len(group) > 0:
#         df = total_df[(total_df[group]).any(axis=1)]
#     else:
#         df = total_df.copy()

#     y_true, y_pred, y_prob = df[target_column], df[prediction_column], df[probability_column]
#     auc = roc_auc_score(y_true, y_prob)

#     results = [auc]
#     methods = [accuracy_score, f1_score, precision_score, recall_score, false_positive_rate]
#     for method in methods:
#         results.append(method(y_true, y_pred))

#     # this is for overall metric, not a subgroup
#     if len(group) == 0:
#         results.extend([None, None])
#         return results

#     # if this is for a subgroup
#     bnsp_auc = compute_bnsp_auc(total_df, group, target_column)
#     bpsn_auc = compute_bpsn_auc(total_df, group, target_column)
#     results.extend([bnsp_auc, bpsn_auc])

#     return results

# def compute_auc(y_true, y_pred):
#     try:
#         return roc_auc_score(y_true, y_pred)
#     except ValueError:
#         return np.nan

# def compute_bpsn_auc(df, subgroup, label):
#     """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
#     subgroup_negative_examples = df[(df[subgroup]).any(axis=1) & (~df[label])]
#     non_subgroup_positive_examples = df[(~df[subgroup]).all(axis=1) & (df[label])]
#     examples = pd.concat([subgroup_negative_examples, non_subgroup_positive_examples])# subgroup_negative_examples.append(non_subgroup_positive_examples)
#     return compute_auc(examples[label], examples[probability_column])

# def compute_bnsp_auc(df, subgroup, label):
#     """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
#     subgroup_positive_examples = df[(df[subgroup]).any(axis=1) & (df[label])]
#     non_subgroup_negative_examples = df[(~df[subgroup]).all(axis=1) & (~df[label])]
#     examples = pd.concat([subgroup_positive_examples, non_subgroup_negative_examples]) # subgroup_positive_examples.append(non_subgroup_negative_examples)
#     return compute_auc(examples[label], examples[probability_column])

# def calculate_overall_auc(df):
#     true_labels = df[target_column]
#     predicted_labels = df[probability_column]
#     return roc_auc_score(true_labels, predicted_labels)

# def power_mean(series, p):
#     total = sum(np.power(series, p))
#     return np.power(total / len(series), 1 / p)

# def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
#     bias_score = np.average([
#         power_mean(bias_df[SUBGROUP_AUC], POWER),
#         power_mean(bias_df[BPSN_AUC], POWER),
#         power_mean(bias_df[BNSP_AUC], POWER)
#     ])
#     return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
