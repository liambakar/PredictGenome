import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv(
    'EE596FinalProject/datasets/TCGA_BRCA_overall_survival_k=0/train.csv')

train_np = train_df.to_numpy()

print(f'The training dataset is of the shape {train_np.shape}')

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

plt.hist(train_df['os_survival_days'])
plt.xlabel('Survival Days')
plt.ylabel('Count')
plt.title('Frequencies of Patients and Survival Times')


plt.subplot(1, 2, 2)

sns.scatterplot(data=train_df, x='age_at_initial_pathologic_diagnosis',
                y='os_survival_days', hue='ajcc_pathologic_tumor_stage', palette='Set2')
plt.title('Scatter Plot Of Time Left to Live')
plt.xlabel('Age of Diagnosis')
plt.ylabel('Survival Days')
plt.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()
