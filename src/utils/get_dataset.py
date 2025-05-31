import os
from typing import Iterable
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.extract_pathway import extract_pathway_summary, load_hallmark_signatures


def get_rna_df():
    base_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'datasets/rna_clean.csv')
    df = pd.read_csv(base_dir)
    # removes last 3 characters from each string in 'sample'
    df['sample'] = df['sample'].apply(lambda x: x[:-3])
    df = df.drop('Unnamed: 0', axis=1)
    return df


def convert_to_32_bit(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        example = df[col].iloc[0]
        if type(example) == np.ndarray:
            df[col] = df[col].apply(lambda x: x.astype(np.float32))
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def get_folded_rna_dataset():
    base_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'clinical_csv')
    cv_folders = [f for f in os.listdir(
        base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    folds = []
    rna_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'datasets/rna_clean.csv')

    Hallmark_pathway_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'datasets/hallmarks_signatures.csv')

    expr = pd.read_csv(rna_path)
    signatures = load_hallmark_signatures(Hallmark_pathway_path)
    sample_ids, sample_paths = extract_pathway_summary(expr, signatures)

    rna_df = pd.DataFrame(sample_paths)
    rna_df.insert(loc=0, column='sample', value=sample_ids)
    rna_df['sample'] = rna_df['sample'].apply(lambda x: x[:-3])

    rna_df.to_csv(os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'datasets/rna_pathways.csv'))

    for folder in cv_folders:
        folder_path = os.path.join(base_dir, folder)
        train_path = os.path.join(folder_path, 'train.csv')
        test_path = os.path.join(folder_path, 'test.csv')

        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"\n=== Processing folder: {folder} ===")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # only use labels and keys
            test_df = test_df[['case_id', 'disc_label']]
            train_df = train_df[['case_id', 'disc_label']]
            # merge rna data with train and test for folds
            train_merged = pd.merge(
                train_df, rna_df, how='inner', left_on='case_id', right_on='sample')
            train_merged = train_merged.drop('sample', axis=1)
            test_merged = pd.merge(
                test_df, rna_df, how='inner', left_on='case_id', right_on='sample')
            test_merged = test_merged.drop('sample', axis=1)

            print(f'Train shape: {train_merged.shape}')
            print(f'Test shape: {test_merged.shape}')

            folds.append({'train': train_merged, 'test': test_merged})

    print(f'\nSuccessfully retrieved {len(folds)} folds.\n')

    return folds


class OnlyGenomicDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], label_col: str):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> tuple[list, float]:
        features = list(self.df.loc[index, self.feature_cols].values)
        label = pd.to_numeric(self.df.loc[index, self.label_col])
        return features, label

    def get_labels(self) -> np.ndarray:
        return self.df[[self.label_col]].values

    def get_features(self) -> np.ndarray:
        return self.df[self.feature_cols].values


def convert_df_to_dataloader(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    batch_size: int,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = OnlyGenomicDataset(df, feature_cols, label_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
