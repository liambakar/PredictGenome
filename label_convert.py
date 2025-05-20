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
    disc_labels, q_bins = compute_discretization(df)
    
    # Add discretized labels as a new column to the DataFrame
    df['disc_label'] = disc_labels
    
    # Create output directory if it doesn't exist
    output_dir = f'/projects/patho5nobackup/TCGA/MMP/EE597/new_labels/TCGA_BRCA_overall_survival_k={fold}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename from the input path
    filename = os.path.basename(path)
    output_path = os.path.join(output_dir, filename)
    
    # Save the modified DataFrame to the new location
    df.to_csv(output_path, index=False)
    
    return df, q_bins

def main():
    folder_number = [0, 1, 2, 3, 4]
    for fold in folder_number:
        cur = f'/projects/patho5nobackup/TCGA/MMP/src/splits/survival/TCGA_BRCA_overall_survival_k={fold}/train.csv'
        load_data(cur, fold)
        cur = f'/projects/patho5nobackup/TCGA/MMP/src/splits/survival/TCGA_BRCA_overall_survival_k={fold}/test.csv'
        load_data(cur, fold)


if __name__ == "__main__":
    main()
