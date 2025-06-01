import pandas as pd
import numpy as np
import os

def compute_discretization(df, survival_time_col='os_survival_days', censorship_col='os_censorship', n_label_bins=4, label_bins=None):
    df = df[~df['case_id'].duplicated()] # make sure that we compute discretization on unique cases

    if label_bins is not None:
        assert len(label_bins) == n_label_bins + 1
        q_bins = label_bins
    else:
        uncensored_df = df[df[censorship_col] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df[survival_time_col], q=n_label_bins, retbins=True, labels=False)
        q_bins[-1] = 1e6  # set rightmost edge to be infinite
        q_bins[0] = -1e-6  # set leftmost edge to be 0

    disc_labels, q_bins = pd.cut(df[survival_time_col], bins=q_bins,
                                retbins=True, labels=False,
                                include_lowest=True)
    assert isinstance(disc_labels, pd.Series) and (disc_labels.index.name == df.index.name)
    disc_labels.name = 'disc_label'
    return disc_labels, q_bins


def load_data(path, fold):
    df = pd.read_csv(path)
    # Compute and add discrete labels, receiving updated df and bins
    df, q_bins = add_discrete_label(df)
    
    # Create output directory if it doesn't exist
    output_dir = f'/home/zhongyuj/why/HW/PredictGenome/new_csv/TCGA_BRCA_overall_survival_k={fold}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename from the input path
    filename = os.path.basename(path)
    output_path = os.path.join(output_dir, filename)
    
    # Save the modified DataFrame to the new location
    df.to_csv(output_path, index=False)
    
    return df, q_bins

def add_discrete_label(df, survival_time_col='dss_survival_days', censorship_col='dss_censorship',
                        n_label_bins=4, new_label_col='disc_label', label_bins=None):
    """
    Add a discrete label column to df based on survival_time_col and censorship_col.
    Returns the modified df and the computed bins.
    """
    # infer censorship column if not provided
    if censorship_col is None:
        prefix = survival_time_col.split('_')[0]
        censorship_col = f"{prefix}_censorship"
    # compute discretized labels and bins
    disc_labels, bins = compute_discretization(df, survival_time_col,
                                              censorship_col,
                                              n_label_bins, label_bins)
    # assign new column
    df[new_label_col] = disc_labels
    return df, bins

def main_combined(n_label_bins=4):
    """
    For each fold, combine train and test data to compute consistent discretization bins,
    then label each set with those bins and save outputs under new_csv.
    """
    folder_numbers = [0, 1, 2, 3, 4]
    for fold in folder_numbers:
        # Load train and test for current fold
        train_path = f'clinical_csv/TCGA_BRCA_overall_survival_k={fold}/train.csv'
        test_path = f'clinical_csv/TCGA_BRCA_overall_survival_k={fold}/test.csv'
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        # Combine to compute shared bins
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        disc_all, q_bins = compute_discretization(df_all,
                                                survival_time_col='dss_survival_days',
                                                censorship_col='dss_censorship',
                                                n_label_bins=n_label_bins)
        # Label train and test with shared bins
        df_train_labeled, _ = add_discrete_label(df_train,
                                                survival_time_col='dss_survival_days',
                                                censorship_col='dss_censorship',
                                                n_label_bins=n_label_bins,
                                                new_label_col='disc_label',
                                                label_bins=q_bins)
        df_test_labeled, _ = add_discrete_label(df_test,
                                                survival_time_col='dss_survival_days',
                                                censorship_col='dss_censorship',
                                                n_label_bins=n_label_bins,
                                                new_label_col='disc_label',
                                                label_bins=q_bins)
        # Save labeled datasets
        output_dir = f'/home/zhongyuj/why/HW/PredictGenome/new_csv/TCGA_BRCA_overall_survival_k={fold}'
        os.makedirs(output_dir, exist_ok=True)
        df_train_labeled.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        df_test_labeled.to_csv(os.path.join(output_dir, 'test.csv'), index=False)


def check_nan_labels(base_dir, n_folds=5, splits=('train','test'), label_col='disc_label'):
    """
    Check for NaN values in the discrete label column across saved CSV files.
    Returns a dict mapping fold -> split -> count of NaNs (only if >0).
    """
    results = {}
    for fold in range(n_folds):
        folder = os.path.join(base_dir, f'TCGA_BRCA_overall_survival_k={fold}')
        for split in splits:
            csv_path = os.path.join(folder, f'{split}.csv')
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path, usecols=[label_col])
            nan_count = df[label_col].isna().sum()
            if nan_count > 0:
                results.setdefault(fold, {})[split] = int(nan_count)
    return results

if __name__ == "__main__":
    # main_combined(n_label_bins=4)
    base = '/home/zhongyuj/why/HW/PredictGenome/new_csv'
    
    nan_results = check_nan_labels(base, n_folds=5, splits=('train', 'test'))
    if nan_results:
        print("Found NaNs in labels:", nan_results)
    else:
        print("No NaNs found in labels across folds and splits.")