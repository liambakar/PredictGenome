import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def get_rna_df():
    base_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'datasets/rna_clean.csv')
    df = pd.read_csv(base_dir)
    # removes last 3 characters from each string in 'sample'
    df['sample'] = df['sample'].apply(lambda x: x[:-3])
    df = df.drop('Unnamed: 0', axis=1)
    return df


def get_folded_rna_dataset():
    base_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))), 'clinical_csv')
    cv_folders = [f for f in os.listdir(
        base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    folds = []

    rna_df = get_rna_df()

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
    def __init__(self, df, feature_cols, label_col):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(
            self.df.loc[index, self.feature_cols].values.tolist(), dtype=torch.float32
        )
        label = torch.tensor(
            self.df.loc[index, self.label_col], dtype=torch.float32
        )
        return features, label

    def get_labels(self) -> torch.Tensor:
        return torch.tensor(self.df[self.label_col].values, dtype=torch.float32)

    def get_features(self) -> torch.Tensor:
        return torch.tensor(self.df[self.feature_cols].values.tolist(), dtype=torch.float32)


def convert_df_to_dataloader(
    df: dict,
    feature_cols:
    list[str],
    label_col: str,
    batch_size: int,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = OnlyGenomicDataset(df, feature_cols, label_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
