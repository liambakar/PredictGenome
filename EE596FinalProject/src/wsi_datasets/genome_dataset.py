import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd


class GenomeDataset(torch.utils.data.Dataset):
    def __init__(self, df_gene, label_bins=None):
        super(GenomeDataset, self).__init__()
        self.df_gene = df_gene
        self.label_bins = label_bins

    def __len__(self):
        return len(self.df_gene)

    def __getitem__(self, idx):
        gene_data = self.df_gene.iloc[idx]
        # Assuming the last column is the label
        return gene_data

    def get_scaler(self):
        numeric_cols = self.df_gene.select_dtypes(
            include=['number']).columns.tolist()
        cols_to_exclude_from_scaling = ['id', 'disc_label']
        numeric_cols_to_scale = [
            col for col in numeric_cols if col not in cols_to_exclude_from_scaling]

        df_numeric = self.df_gene[numeric_cols_to_scale]

        scaler = StandardScaler().fit(df_numeric)
        return scaler

    def apply_scaler(self, scaler):
        numeric_cols = self.df_gene.select_dtypes(
            include=['number']).columns.tolist()
        cols_to_exclude_from_scaling = ['id', 'disc_label']
        numeric_cols_to_scale = [
            col for col in numeric_cols if col not in cols_to_exclude_from_scaling]

        df_numeric = self.df_gene[numeric_cols_to_scale]
        df_non_numeric = self.df_gene.drop(columns=numeric_cols_to_scale)

        scaled_data = scaler.transform(df_numeric)

        scaled_df = pd.DataFrame(
            scaled_data, columns=numeric_cols_to_scale, index=self.df_gene.index)

        self.df_gene = df_non_numeric.join(scaled_df)

    def get_label_bins(self):
        return self.label_bins


def build_datasets(csv_splits, batch_size=1, num_workers=2,
                   train_kwargs={'shuffle': True},
                   val_kwargs={'shuffle': False}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    scaler = None
    print('\n')
    for k in csv_splits.keys():  # ['train', 'val', 'test']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = GenomeDataset(df_gene=df)

        if k == 'train':
            scaler = dataset.get_scaler()
            assert scaler is not None, "Scaler from train split required"
        dataset.apply_scaler(scaler)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=dataset_kwargs['shuffle'], num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
        if k == 'train':
            label_bins = dataset.get_label_bins()
    return dataset_splits
