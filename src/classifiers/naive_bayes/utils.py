from difflib import SequenceMatcher
import pandas as pd


def prepare_dataset_based_on_class(df: pd.DataFrame, y_label: str,
                                   y_classes: list) -> pd.DataFrame:
    all_possibles_classes = df[y_label].unique()
    if len(all_possibles_classes) == len(y_classes):
        return df
    elif len(y_classes) < 2:
        print(
            "Minimal number of analyzed class is 2, setting analyzed class "
            "into two -> positive and negative")
        y_classes = ['negative', 'positive']
    class_to_delete = set(all_possibles_classes) - set(y_classes)
    df = df.loc[df[y_label] != list(class_to_delete)[0]].reset_index(drop=True)
    return df


def remove_junk_transcriptions(df_row):
    s = SequenceMatcher(None, df_row['Utterance'], df_row['Transcription'])
    return s.ratio()


def process_data(df: pd.DataFrame, y_label: str, y_classes: list,
                 match_threshold: float):
    df = df.dropna()
    df = df[df.apply(lambda row: remove_junk_transcriptions(row),
                     axis=1) > match_threshold]
    df = prepare_dataset_based_on_class(df, y_label=y_label,
                                        y_classes=y_classes)
    return df
