from pathlib import Path
import pandas as pd

# adjust to wherever your fold-dirs live
root = Path('../clinical_csv')
all_dfs = []

# 1) read everything in
for k in range(5):
    fold = root / f'TCGA_BRCA_overall_survival_k={k}'
    for split in ('train', 'test'):
        csv_file = fold / f'{split}.csv'
        if not csv_file.exists():
            print(f"⚠️ Missing: {csv_file}")
            continue

        df = pd.read_csv(csv_file, usecols=['case_id', 'disc_label'])
        df['fold']  = k
        df['split'] = split
        all_dfs.append(df)

big_df = pd.concat(all_dfs, ignore_index=True)

# find all case_ids with >1 distinct non-null disc_label
conflict_labels = (
    big_df
    .groupby('case_id')['disc_label']
    .apply(lambda x: sorted(set(x.dropna())))   # collect unique non-NaN labels
    .loc[lambda s: s.map(len) > 1]              # keep only those with more than one label
)

if conflict_labels.empty:
    print("✅ No conflicting labels found.")
else:
    print("⚠️ Conflicts detected:")
    for case_id, labels in conflict_labels.items():
        print(f"{case_id}: {labels}")

# 3) dedupe, preferring non-NaN labels:
#    add a helper column so that rows with NaN labels sort *after* those with real labels
cleaned = (
    big_df
    .assign(label_null=big_df['disc_label'].isna())
    .sort_values(['case_id', 'label_null'])        # non-null (False) comes before null (True)
    .drop_duplicates(subset='case_id', keep='first')
    .drop(columns='label_null')
)

# 4) save
out_dir  = Path('../datasets')
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / 'all_case_ids_disc_labels.csv'
cleaned.to_csv(out_file, index=False)

print(f"✅ Saved {len(cleaned):,} unique case_ids to {out_file}")
