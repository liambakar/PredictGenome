import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from baseline_clinical_r import feature_selection, preprocess_features


def convert_to_32_bit(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        example = df[col].iloc[0]
        if type(example) == np.ndarray:
            df[col] = df[col].apply(lambda x: x.astype(np.float32))
        elif type(example) == list:
            df[col] = df[col].apply(lambda x: [np.float32(x_i) for x_i in x])
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def get_rna_dataset(json_dataset_path: str):
    json_dataset_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), json_dataset_path)
    df = pd.read_json(json_dataset_path, lines=True)
    feature_cols = list(df.columns[1:-1])
    return df, feature_cols


def get_rna_folds(json_dataset_path: str = 'datasets/final_dataset.json', k_splits: int = 5, shuffle: bool = True) -> list[dict[str, pd.DataFrame]]:
    df, _ = get_rna_dataset(json_dataset_path)
    kf = StratifiedKFold(n_splits=k_splits, shuffle=shuffle)
    y = df['disc_label']
    x = df.drop('disc_label', axis=1)
    x = x.apply(lambda x: np.array(x))
    folds = []
    for i, (train_i, test_i) in enumerate(kf.split(x, y)):

        print(f"\n==== Fold {i} ====")

        train_df = df.iloc[train_i]
        test_df = df.iloc[test_i]

        print(f'Train shape: {train_df.shape}')
        print(f'Test shape: {test_df.shape}')

        folds.append({'train': train_df, 'test': test_df})

    print(f'\nSuccessfully retrieved {len(folds)} folds.\n')
    return folds


def get_all_clinical_data(path_to_clinical: str):
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        path_to_clinical
    )
    cv_folders = []
    for f in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, f)
        if os.path.isdir(folder_path):
            cv_folders.append(folder_path)

    dfs = []
    for f in cv_folders:
        train_path = os.path.join(f, 'train.csv')
        test_path = os.path.join(f, 'test.csv')
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            dfs.append(train_df)
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            dfs.append(test_df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates()

    return combined_df


def merge_clinical_and_genomic(
        clinical_df: pd.DataFrame,
        genomic_df: pd.DataFrame
) -> tuple[pd.DataFrame, list]:
    clinical_cols = feature_selection(clinical_df, multimodal=True)
    clinical_features, _, _, _ = preprocess_features(
        clinical_df,
        clinical_cols
    )

    merged = pd.merge(
        left=genomic_df,
        right=clinical_features,
        how='left',
        left_on='sample_id',
        right_on='case_id',
    )
    merged = merged.drop(columns=['case_id'])
    clinical_cols.remove('case_id')
    return merged, clinical_cols


def get_clinical_and_genomic_data(
    path_to_clinical: str = 'clinical_csv_consistent',
    json_dataset_path: str = 'datasets/final_dataset.json',
) -> tuple[pd.DataFrame, list, list]:
    rna_df, rna_feature_cols = get_rna_dataset(json_dataset_path)
    print('Retrieved Genomic Dataset.')
    clinical_df = get_all_clinical_data(path_to_clinical)
    print('Retrieved Clinical Dataset.')
    merged, clinical_cols = merge_clinical_and_genomic(clinical_df, rna_df)
    print('Merged Genomic and Clinical Datasets.')
    return merged, clinical_cols, rna_feature_cols


def get_merged_folds(df, k_splits=5, shuffle=True) -> list[dict[str, pd.DataFrame]]:
    kf = StratifiedKFold(n_splits=k_splits, shuffle=shuffle)
    y = df['disc_label']
    x = df.drop('disc_label', axis=1)
    x = x.apply(lambda x: np.array(x))
    folds = []
    for i, (train_i, test_i) in enumerate(kf.split(x, y)):

        print(f"\n==== Fold {i} ====")

        train_df = df.iloc[train_i]
        test_df = df.iloc[test_i]

        print(f'Train shape: {train_df.shape}')
        print(f'Test shape: {test_df.shape}')

        folds.append({'train': train_df, 'test': test_df})

    print(f'\nSuccessfully retrieved {len(folds)} folds.\n')
    return folds
