from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd

prediction_column = 'preds'
probability_column = 'probs'
target_column = 'labels'
id_column = 'id' # result csv files always has id in this same column. can be different in train or test csv files

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

def calculate_demographic_parity(df_privileged, df_unprivileged):
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

def calculate_sensitive_accuracy(df_privileged, df_unprivileged):
    accuracy_privileged = accuracy_score(df_privileged[target_column], df_privileged[prediction_column])
    accuracy_unprivileged = accuracy_score(df_unprivileged[target_column], df_unprivileged[prediction_column])

    accuracy = (accuracy_privileged + accuracy_unprivileged) / 2
    return accuracy_unprivileged, accuracy_privileged, accuracy

def calculate_bias(df, privileged_group, unprivileged_group):
    df_privileged = df[(df[privileged_group]).any(axis=1)]
    df_unprivileged = df[(df[unprivileged_group]).any(axis=1)]

    demographic_parity = calculate_demographic_parity(df_privileged, df_unprivileged)
    EqOpp1, EqOpp0, EqOdd = calculate_equality(df_privileged, df_unprivileged)
    accuracy_unprivileged, accuracy_privileged, accuracy = calculate_sensitive_accuracy(df_privileged, df_unprivileged)

    biases = [demographic_parity, EqOpp1, EqOpp0, EqOdd, accuracy_unprivileged, accuracy_privileged, accuracy]
    return biases



def calculate_metrics(df, group):
    # if group is empty return the overall metric. 
    # otherwise only calculate for example of that group
    if len(group) > 0:
        df = df[(df[group]).any(axis=1)]

    y_true, y_pred, y_prob = df[target_column], df[prediction_column], df[probability_column]
    auc = roc_auc_score(y_true, y_prob)

    results = [auc]
    methods = [accuracy_score, f1_score, precision_score, recall_score, false_positive_rate]
    for method in methods:
        results.append(method(y_true, y_pred))

    return results