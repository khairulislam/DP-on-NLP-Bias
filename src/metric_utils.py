from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

prediction_column = 'preds'
probability_column = 'probs'
target_column = 'labels'
id_column = 'id' # result csv files always has id in this same column. can be different in train or test csv files

def get_overall_results(result:pd.DataFrame, groups:list[str]):
    overall_results = {
        'metrics': ['auc', 'accuracy', 'f1_score', 
        'precision', 'recall', 'false positive rate']
    }

    for group in groups:
        overall_results[group] = calculate_metrics(result, group)

    overall_results['Total'] = calculate_metrics(result, None)

    overall_results = pd.DataFrame(overall_results)
    return overall_results

def get_identity_count(train_df:pd.DataFrame, test_df:pd.DataFrame, identities:list[str]):
    count_dict = {
        'Identity':identities,
        '0 (train)':[],
        '1 (train)':[],
        '0 (test)':[],
        '1 (test)':[],
    }
    for identity in identities:
        train_neg, train_pos = train_df[train_df[identity]>=0.5][target_column].value_counts().to_numpy()
        test_neg, test_pos = test_df[test_df[identity]>=0.5][target_column].value_counts().to_numpy()
        count_dict['0 (train)'].append(train_neg)
        count_dict['1 (train)'].append(train_pos)
        count_dict['0 (test)'].append(test_neg)
        count_dict['1 (test)'].append(test_pos)

    return pd.DataFrame(count_dict)

def binarize(df, columns):
    for col in columns:
        df[col] = (df[col] >= 0.5)
    return df

# https://en.wikipedia.org/wiki/Confusion_matrix
def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    negatives = fp+tn
    if negatives == 0:
        print('Warning ! No negative examples found')
        false_positive_rate = np.nan
    else: 
        false_positive_rate = fp / negatives
    return false_positive_rate

def true_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    positives = fn + tp
    if positives == 0:
        print('Warning ! No positive examples found')
        true_positive_rate = np.nan
    else: 
        true_positive_rate = tp / positives
    return true_positive_rate

def calculate_demographic_parity(df_privileged:pd.DataFrame, df_unprivileged:pd.DataFrame):
    privileged_number = df_privileged.shape[0]
    if privileged_number >0:
        privileged_positive_percent = sum(df_privileged[prediction_column]) / privileged_number
    else:
        print('Warning, no privileged example found')
        privileged_positive_percent = 0

    unprivileged_number = df_privileged.shape[0]
    if unprivileged_number >0:
        unprivileged_positive_percent = sum(df_unprivileged[prediction_column]) / unprivileged_number
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

def positiveAEG(df:pd.DataFrame, subgroup:str):
    subgroup_positive_examples = df[df[subgroup] & (df[target_column])]
    background_positive_examples = df[(~df[subgroup]) & (df[target_column])]
    stat, p_value = mannwhitneyu(subgroup_positive_examples[prediction_column], background_positive_examples[prediction_column])
    return 0.5 - stat*1.0/(len(background_positive_examples)*len(subgroup_positive_examples))

def negativeAEG(df:pd.DataFrame, subgroup:str):
    subgroup_negative_examples = df[df[subgroup] & (~df[target_column])]
    background_negative_examples = df[(~df[subgroup]) & (~df[target_column])]
    stat, p_value = mannwhitneyu(subgroup_negative_examples[prediction_column], background_negative_examples[prediction_column])
    # print(f'{subgroup} Negative {stat, p_value}')
    return 0.5 - stat*1.0/(len(background_negative_examples)*len(subgroup_negative_examples))

def get_all_bias(df:pd.DataFrame, protected_groups:list[str]):
    bias_results = {
        'fairness_metrics': ['parity', 'eqOpp1',
        'eqOpp0', 'eqOdd', 'p-accuracy',
        'auc', 'bnsp', 'bpsn', 'posAEG', 'negAEG']
        }

    for group in protected_groups:
        bias_results[group] = calculate_bias(df, group)

    return pd.DataFrame(bias_results)

def calculate_bias(df:pd.DataFrame, group:str):
    df_protected = df[df[group]]
    df_background = df[~df[group]]

    demographic_parity = calculate_demographic_parity(df_background, df_protected)
    eqOpp1, eqOpp0, eqOdd = calculate_equality(df_background, df_protected)
    accuracy_protected = accuracy_score(df_protected[target_column], df_protected[prediction_column])

    biases = [demographic_parity, eqOpp1, eqOpp0, eqOdd, accuracy_protected]

    auc = roc_auc_score(df_protected[target_column], df_protected[probability_column])
    bnsp_auc = compute_bnsp_auc(df, group, target_column)
    bpsn_auc = compute_bpsn_auc(df, group, target_column)
    posAEG = positiveAEG(df, group)
    negAEG = negativeAEG(df, group)

    biases.extend([auc, bnsp_auc, bpsn_auc, posAEG, negAEG])
    return biases

SUBGROUP_AUC = 'auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def calculate_metrics(total_df:pd.DataFrame, group:str=None):
    # if group is None return the overall metric. 
    # otherwise only calculate for example of that group
    if group:
        df = total_df[total_df[group]]
    else:
        df = total_df.copy()

    y_true, y_pred, y_prob = df[target_column], df[prediction_column], df[probability_column]
    auc = roc_auc_score(y_true, y_prob)

    results = [auc]
    methods = [accuracy_score, f1_score, precision_score, recall_score, false_positive_rate]
    for method in methods:
        results.append(method(y_true, y_pred))

    return results

def compute_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_bpsn_auc(df, subgroup, label):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & (~df[label])]
    non_subgroup_positive_examples = df[(~df[subgroup]) & (df[label])]
    examples = pd.concat([subgroup_negative_examples, non_subgroup_positive_examples])# subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[probability_column])

def compute_bnsp_auc(df, subgroup, label):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & (df[label])]
    non_subgroup_negative_examples = df[(~df[subgroup]) & (~df[label])]
    examples = pd.concat([subgroup_positive_examples, non_subgroup_negative_examples]) # subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[probability_column])

def calculate_overall_auc(df):
    true_labels = df[target_column]
    predicted_labels = df[probability_column]
    return roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)