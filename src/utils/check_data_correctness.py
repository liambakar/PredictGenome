import pandas as pd
from pathlib import Path

def get_disc_label(root_dir, target_case_id, check_label):
    root = Path(root_dir)
    result = []

    for k_folder in root.glob("TCGA_BRCA_overall_survival_k=*"):
        for split in ['train.csv', 'test.csv']:
            csv_file = k_folder / split
            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file)
            if 'case_id' not in df.columns or check_label not in df.columns:
                continue

            matched = df[df['case_id'] == target_case_id]
            if not matched.empty:
                result.append({
                    'file': str(csv_file),
                    'case_id': target_case_id,
                    check_label: matched.iloc[0][check_label]
                })

    return result

# Example usage
root_directory = '../clinical_csv_dss'
case_id = 'TCGA-OL-A5D7'
check_label = 'disc_label'
# check_label = 'os_survival_days'
results = get_disc_label(root_directory, case_id, check_label)

for res in results:
    print(res)
