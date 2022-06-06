import pandas as pd
from dataclasses import dataclass

@dataclass
class Config:
    model_name = 'bert-base-uncased'
    dataset_name = 'social_bias_frames'
    text_column = 'post'
    # if id column is string, replace that with an integer index during preprocessing
    raw_id_column = 'HITId'
    id_column = 'index'
    need_to_split = False

    # target in raw dataset. However, it will be renamed to `labels` here to facilitate training setup
    raw_target_column = 'offensiveYN'
    target_column = 'labels'
    # test and validation data with each be 50% of this amount
    test_size = 0.3
    max_seq_length = 128
    seed = 2022

text_column = Config.text_column
target_column = Config.raw_target_column
id_column = Config.raw_id_column
identities = ['male', 'female', 'white', 'black']
features = [text_column, target_column] + identities + ['targetMinority', 'targetCategory']
selected_columns = [id_column] + features

class SocialBiasProcessor:
    def __init__(self) -> None:
        pass

    def is_male(word: str):
        word = word.lower().strip()
        if 'trans' in word: return False

        for w in ['male', 'men', 'man']:
            if word.startswith(w) or ' '+ w in word:
                return True

        for w in ['father', 'boy', 'incel']:
            if w in word:
                return True
        return False

    def is_female(word: str):
        word = word.lower().strip()
        if 'trans' in word: return False
        
        for w in ['female', 'women', 'woman', 'mother', 'lesbian', 'girl']:
            if w in word:
                return True
        return False

    def is_white(word: str):
        word = word.lower().strip()
        for neg in ['not', 'non']:
            for add in ['', ' ', '-']:
                if neg + add + 'white' in word:
                    return False

        return 'white' in word

    def is_black(word: str):
        word = word.lower().strip()
        for neg in ['not', 'non']:
            for add in ['', ' ', '-']:
                if (neg + add + 'black' in word) or (neg + add + 'dark' in word):
                    return True

        return ('black' in word) or ('dark' in word)

    @staticmethod
    def process(df: pd.DataFrame):
        # create binary columns for target indentity groups from annotation
        categories = ['gender', 'gender', 'race', 'race']
        functions = [SocialBiasProcessor.is_male, SocialBiasProcessor.is_female, SocialBiasProcessor.is_white, SocialBiasProcessor.is_black]
        for index, column in enumerate(identities):
            df[column] = df[df['targetCategory']==categories[index]]['targetMinority'].apply(functions[index])
            df[column].fillna(False, inplace=True)

        # ensure text is in string format
        df.loc[:, text_column] = df[text_column].astype(str) 
        # convert target label to binary
        df.loc[:, target_column] = df[target_column].apply(lambda x: 1 if x!='' and float(x)>=0.5 else 0)

        # https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value
        grouped = df.groupby([id_column])[identities].agg('mean').reset_index()
        for identity in identities:
            grouped[identity] = grouped[identity].apply(lambda x: 1 if x >= 0.5 else 0)

        df_unique = df.drop_duplicates(subset=id_column)[[id_column, text_column, target_column]]
        df = df_unique.merge(grouped, on=id_column, how='inner').reset_index(drop=True)
        return df