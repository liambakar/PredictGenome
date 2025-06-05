from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Any


class PDataset(ABC, Dataset):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> Any:
        pass

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_features(self) -> Any:
        pass

    @abstractmethod
    def get_dataloader(self, batch_size, shuffle) -> DataLoader:
        pass


class OnlyGenomicDataset(PDataset):

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], label_col: str):
        self.df = df.copy()
        self.df.reset_index(drop=True)
        for col in self.df.columns:
            if isinstance(self.df[col].values[0], (list, tuple)):
                self.df[col] = self.df[col].apply(np.array)
        self.feature_cols = feature_cols
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> tuple[list, torch.Tensor]:
        row = self.df.iloc[index]
        features = list(row.loc[self.feature_cols].values)
        for i, f in enumerate(features):
            features[i] = torch.tensor(f, dtype=torch.float32)

        label = torch.tensor(pd.to_numeric(
            row[self.label_col], downcast='float'), dtype=torch.float32)
        return features, label

    def get_labels(self) -> np.ndarray:
        return self.df[[self.label_col]].values

    def get_features(self) -> np.ndarray:
        return self.df[self.feature_cols].values

    def get_dataloader(self, batch_size, shuffle=True):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        return dataloader


class GenomicAndClinicalDataset(PDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        genomic_feature_cols: list[str],
        clinical_feature_cols: list[str],
        label_col: str,
    ):
        self.df = df.copy()
        self.df.reset_index(drop=True)
        for col in self.df.columns:
            if isinstance(self.df[col].values[0], (list, tuple)):
                self.df[col] = self.df[col].apply(np.array)
        self.genomic_feature_cols = genomic_feature_cols
        self.clinical_feature_cols = clinical_feature_cols
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> tuple[list, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        genomic_features = list(row.loc[self.genomic_feature_cols].values)
        for i, f in enumerate(genomic_features):
            genomic_features[i] = torch.tensor(f, dtype=torch.float32)
        clinical_features = torch.tensor(
            row.loc[self.clinical_feature_cols])

        label = torch.tensor(pd.to_numeric(
            row[self.label_col], downcast='float'), dtype=torch.float32)
        return genomic_features, clinical_features, label

    def get_labels(self) -> np.ndarray:
        return self.df[[self.label_col]].values

    def get_features(self) -> tuple[np.ndarray, np.ndarray]:
        return self.df[self.genomic_feature_cols].values, self.df[self.clinical_feature_cols].values

    def get_dataloader(self, batch_size, shuffle=True):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        return dataloader
