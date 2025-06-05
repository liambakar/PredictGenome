import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from lifelines.utils import concordance_index
import numpy as np

base_dir = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'clinical_csv')
cv_folders = [f for f in os.listdir(
    base_dir) if os.path.isdir(os.path.join(base_dir, f))]

results = []
cidx_list = []


def feature_selection(df, multimodal=False):
    selected_features = [
        'cancer_type_detailed',
        # 'tissue_source_site',
        'OncoTreeSiteCode',
        'age_at_initial_pathologic_diagnosis',
        'race',
        'ajcc_pathologic_tumor_stage',
        'clinical_stage',
        'histological_type',
        'histological_grade',
        'initial_pathologic_dx_year',
        'menopause_status',
        # 'birth_days_to',
        # 'vital_status',
        # 'tumor_status',
        'margin_status'
    ]
    if multimodal:
        selected_features.insert(0, 'case_id')
    selected_features = [col for col in selected_features if col in df.columns]
    return selected_features


def preprocess_features(df, feature_cols):
    """
    Preprocess feature data
    """
    processed_df = df[feature_cols].copy()

    # Process categorical variables
    categorical_cols = []
    numerical_cols = []

    for col in feature_cols:
        if col in processed_df.columns:
            if (processed_df[col].dtype == 'object' or processed_df[col].dtype.name == 'category') \
                    and col != 'case_id':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values
        processed_df[col] = processed_df[col].fillna('Unknown')
        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        label_encoders[col] = le

    # Handle missing values in numerical variables
    for col in numerical_cols:
        if col != 'case_id':
            processed_df[col] = processed_df[col].fillna(
                processed_df[col].median())

    return processed_df, label_encoders, categorical_cols, numerical_cols


def main():

    # Main processing logic
    for folder in cv_folders:
        folder_path = os.path.join(base_dir, folder)
        train_path = os.path.join(folder_path, 'train.csv')
        test_path = os.path.join(folder_path, 'test.csv')

        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"\n=== Processing folder: {folder} ===")

            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print("Train shape:", train_df.shape)
            print("Test shape:", test_df.shape)

            # Feature selection
            selected_features = feature_selection(train_df)

            print(f"\nFeature selection results:")
            print(f"- Total feature count: {len(selected_features)}")
            print(f"\nLabel (disc_label) distribution:")
            print(train_df['disc_label'].value_counts())

            # Preprocess features
            X_train, label_encoders, cat_cols, num_cols = preprocess_features(
                train_df, selected_features)
            X_test, _, _, _ = preprocess_features(test_df, selected_features)
            y_train = train_df['disc_label']
            y_test = test_df['disc_label'] if 'disc_label' in test_df.columns else None

            # print(f"\nPost-processing:")
            # print(f"X_train shape: {X_train.shape}")
            # print(f"X_test shape: {X_test.shape}")
            # print(f"Categorical feature count: {len(cat_cols)}")
            # print(f"Numerical feature count: {len(num_cols)}")

            # # Data quality check
            # print(f"\nData quality:")
            # print(f"Training set missing values: {X_train.isnull().sum().sum()}")
            # print(f"Test set missing values: {X_test.isnull().sum().sum()}")

            # Remove samples with NaN labels
            mask_train = ~y_train.isnull()
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]

            if y_test is not None:
                mask_test = ~y_test.isnull()
                X_test = X_test[mask_test]
                y_test = y_test[mask_test]

            # Save processing results to dictionary for later use
            fold_result = {
                'folder': folder,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': selected_features,
                'label_encoders': label_encoders,
                'categorical_features': cat_cols,
                'numerical_features': num_cols
            }

            results.append(fold_result)

            for i, fold_data in enumerate(results):
                print(f"Training fold {i+1}")

                X_train = fold_data['X_train']
                X_test = fold_data['X_test']
                y_train = fold_data['y_train']
                y_test = fold_data['y_test']

                # XGBoost model
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    random_state=42
                )

                model.fit(X_train, y_train)
                # Predict probabilities
                y_pred_proba = model.predict_proba(X_test)
                # Take the probability of the predicted maximum probability class as risk score
                high_risk_class_index = int(y_train.max())
                risk_score = y_pred_proba[:, high_risk_class_index]
                # Calculate c-index (use risk_score and true labels)
                if y_test is not None and len(np.unique(y_test)) > 1:
                    cidx = concordance_index(y_test, risk_score)
                    print(f"c-index: {cidx:.4f}")
                    cidx_list.append(cidx)
                else:
                    print("Cannot compute c-index (not enough classes in y_test)")

    mean_cidx = np.mean(cidx_list)
    std_cidx = np.std(cidx_list)
    print(f"\nFinal c-index: {mean_cidx:.4f} ± {std_cidx:.4f}")

    if __name__ == "__main__":
        main()
