## @package demo_QHCC_TabPFN
# @version v1.1
# @brief The demo file for predicting recurrence in QHCC datasets using TabPFN
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tabpfn import TabPFNClassifier
from pymfe.mfe import MFE  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '1'

## @brief Feature selection to select the top k features with the highest mutual information values
# Input: 
# @param  X0: dataframe, the tabular data
# @param  y0: dataframe, the label data
# @param  k: integer, the number of feature elements to be selected
# Output: 
# @return selected_features: the indexes of the selected features
def auto_feature_select(X0, y0, k=1):
    
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X0, y0)

    ## Indices of the selected features
    selected_features = selector.get_support(indices=True)

    return selected_features

## @brief Finds the best probability threshold for binary classification to maximize the F1-score.
# Input:
# @param y_true np.array: true binary labels
# @param y_probs np.array: predicted probabilities for the positive class
#Output:
# @return best_threshold float: the probability threshold that maximizes the F1-score
# @return best_f1_score float: the maximum F1-score achieved at the best_threshold    
def find_best_threshold_by_f1(y_true, y_probs):

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1
    
## @brief Experiments on prediction with tabular data
# Input: 
# @param  df: dataframe, the tabular data
# @param  seed: random seed
# Output: 
# @return model: the trained model
# @return ids_train: the patient IDs for the training data 
# @return ids_test: the patient IDs for the test data 
# @return acc: accuracy 
# @return auc: AUC (area under the curve) score
def experiment_redcap_lab(df, ids_train, ids_test, seed = 100): 

    ## Copy data
    new_df = df.copy()
    ids_df = df["QHCCID"]

    ## Build features and labels
    X_df = new_df.drop("Recurrence", axis=1)
    y_df = new_df[["Recurrence"]]

    ## Use label encoding to convert text labels to numeric data and fill missing values with 'unknown'
    label_encoder = LabelEncoder()
    for column in X_df.select_dtypes(include=['object']).columns:
        X_df[column] = label_encoder.fit_transform(X_df[column].fillna('unknown'))

    y_df.loc[:, 'Recurrence'] = y_df['Recurrence'].fillna(0)
    y_df.loc[:, 'label'] = y_df['Recurrence'].astype(int)

    ## Create a label-to-name dictionary for easy observation of results
    label_to_name = {0: "No Recurrence", 1: "Recurrence"}

    ## Create an imputer to fill missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_df)

    ## Extract label data
    y = y_df["label"]
    n_class = len(np.unique(y))
    print(f'The number of classes: {n_class}')

    ## Split data following the previous split of patient groups
    X_train = X_df.loc[[np.isin(i, ids_train).item() for i in ids_df ] ]
    X_test = X_df.loc[[np.isin(i, ids_test).item() for i in ids_df ] ]
    y_train = y.loc[[np.isin(i, ids_train).item() for i in ids_df ] ]
    y_test = y.loc[[np.isin(i, ids_test).item() for i in ids_df ] ]
    
    ## Extract meta-features from the original training set (before further processing)
    mfe = MFE(groups=["general", "statistical", "info-theory", "model-based"])
    mfe.fit(X_train.to_numpy(), y_train.to_numpy())  # Ensure input is NumPy array
    ft_names, ft_values = mfe.extract()
    metafeatures = dict(zip(ft_names, ft_values))

    print("Extracted Metafeatures:", metafeatures.keys())

    # Calculate missing values (manually, bypassing PyMFE)
    missing_count_train = X_train.isnull().sum().sum()
    missing_count_test = X_test.isnull().sum().sum()
    print(f"Total missing values (train): {missing_count_train}, (test): {missing_count_test}")

    # Fill missing values
    if missing_count_train > 0 or missing_count_test > 0:
        print("Filling missing values...")
        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Normalization / Standardization
    if metafeatures.get("skewness.mean", 0) > 1.0:
        print("Applying StandardScaler due to skewness...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif metafeatures.get("kurtosis.mean", 0) > 3:
        print("Applying MinMaxScaler due to high kurtosis...")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Automatic feature selection using RFECV + SelectKBest(mutual_info)
    print("Running RFECV to determine optimal number of features...")
    counter = Counter(y_train)
    neg, pos = counter[0], counter[1]
    scale_pos_weight = neg / pos if pos > 0 else 1.0  # 防止除以 0

    rfecv_estimator = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(estimator=rfecv_estimator, step=1, cv=cv, scoring="f1")
    rfecv.fit(X_train, y_train)
    optimal_k = rfecv.n_features_
    print(f"Optimal number of features selected by RFECV: {optimal_k}")

    selected_feature_indices = auto_feature_select(X_train, y_train, k=optimal_k)
    X_train = X_train.iloc[:, selected_feature_indices]
    X_test = X_test.iloc[:, selected_feature_indices]

    # Dimensionality reduction (PCA)
    if metafeatures.get("cor.mean", 0) > 0.9:
        print("Applying PCA due to high correlation...")
        pca = PCA(n_components=10)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Class imbalance handling (SMOTE)
    if metafeatures.get("class_ent", 0) < 0.5:
        print("Applying SMOTE due to class imbalance...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Initialize TabPFNClassifier
    model = TabPFNClassifier()

    ## Train TabPFNClassifier
    model.fit(X_train, y_train)

    ## Make prediction
    y_probs = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = find_best_threshold_by_f1(y_test, y_probs)
    print(f"Optimal threshold by F1: {best_thresh:.2f}, F1: {best_f1:.4f}")

    y_pred = (y_probs > best_thresh).astype(int)

    ## Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    print('Prediction with table data:')
    print(f'Accuracy: {acc * 100:.2f}%')
    print(f'AUC: {auc * 100:.2f}%')
    ## Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[label_to_name[item] for item in set(y_test)]))
    
    return model, ids_train, ids_test, acc, auc

## @brief Experiments on prediction with both tabular data and radiomics features
# Input: 
# @param  df: dataframe, the tabular data 
# @param  df_1: dataframe, the radiomics features extracted from MRI images
# @param  df_2: dataframe, the radiomics features extracted from CT images
# @param  ids_train: the patient IDs for the training data 
# @param  ids_test: the patient IDs for the test data 
# @param  seed: random seed
# Output: 
# @return  model: the trained model
# @return  ids_train: the patient IDs for the training data 
# @return  ids_test: the patient IDs for the test data 
# @return  acc: accuracy 
# @return  auc: AUC (area under the curve) score 
def experiment_redcap_lab_radiomics(new_df, ids_train=None, ids_test=None, seed = 100): 

    ## Remove the QHCCID column 
    id_new_df = new_df["QHCCID"] 
    new_df = new_df.drop("QHCCID", axis=1)

    ## Build features and labels
    new_X_df = new_df.drop("Recurrence", axis=1)
    new_y_df = new_df[["Recurrence"]]

    label_encoder = LabelEncoder()
    for column in new_X_df.select_dtypes(include=['object']).columns:
        new_X_df[column] = label_encoder.fit_transform(new_X_df[column].fillna('unknown'))

    new_y_df.loc[:, 'Recurrence'] = new_y_df['Recurrence'].fillna(0)
    new_y_df.loc[:, 'label'] = new_y_df['Recurrence'].astype(int)

    new_label_to_name = {0: "No Recurrence", 1: "Recurrence"}

    imputer = SimpleImputer(strategy='mean')
    new_X_imputed = imputer.fit_transform(new_X_df)
    new_X_imputed.shape

    ## Copy labels
    y = new_y_df["label"]
    n_class = len(np.unique(y))
    print(f'The number of classes: {n_class}')

    ## Split data following the previous split of patient groups
    X_train = new_X_df.loc[[np.isin(i, ids_train).item() for i in id_new_df ] ]
    X_test = new_X_df.loc[[np.isin(i, ids_test).item() for i in id_new_df ] ]
    y_train = y.loc[[np.isin(i, ids_train).item() for i in id_new_df ] ]
    y_test = y.loc[[np.isin(i, ids_test).item() for i in id_new_df ] ]
    
    ## Extract meta-features from the original training set (before further processing)
    mfe = MFE(groups=["general", "statistical", "info-theory", "model-based"])
    mfe.fit(X_train.to_numpy(), y_train.to_numpy())  # Ensure input is NumPy array
    ft_names, ft_values = mfe.extract()
    metafeatures = dict(zip(ft_names, ft_values))

    print("Extracted Metafeatures:", metafeatures.keys())

    # Calculate missing values (manually, bypassing PyMFE)
    missing_count_train = X_train.isnull().sum().sum()
    missing_count_test = X_test.isnull().sum().sum()
    print(f"Total missing values (train): {missing_count_train}, (test): {missing_count_test}")

    # Fill missing values
    if missing_count_train > 0 or missing_count_test > 0:
        print("Filling missing values...")
        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Normalization / Standardization
    if metafeatures.get("skewness.mean", 0) > 1.0:
        print("Applying StandardScaler due to skewness...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif metafeatures.get("kurtosis.mean", 0) > 3:
        print("Applying MinMaxScaler due to high kurtosis...")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Automatic feature selection using RFECV + SelectKBest(mutual_info)
    print("Running RFECV to determine optimal number of features...")
    counter = Counter(y_train)
    neg, pos = counter[0], counter[1]
    scale_pos_weight = neg / pos if pos > 0 else 1.0  # 防止除以 0

    rfecv_estimator = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(estimator=rfecv_estimator, step=1, cv=cv, scoring="f1")
    rfecv.fit(X_train, y_train)
    optimal_k = rfecv.n_features_
    print(f"Optimal number of features selected by RFECV: {optimal_k}")

    selected_feature_indices = auto_feature_select(X_train, y_train, k=optimal_k)
    X_train = X_train.iloc[:, selected_feature_indices]
    X_test = X_test.iloc[:, selected_feature_indices]

    # Dimensionality reduction (PCA)
    if metafeatures.get("cor.mean", 0) > 0.9:
        print("Applying PCA due to high correlation...")
        pca = PCA(n_components=10)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Class imbalance handling (SMOTE)
    if metafeatures.get("class_ent", 0) < 0.5:
        print("Applying SMOTE due to class imbalance...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Initialize TabPFNClassifier
    model = TabPFNClassifier()

    ## Train TabPFNClassifier
    model.fit(X_train, y_train)

    ## Make predictions
    y_probs = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = find_best_threshold_by_f1(y_test, y_probs)
    print(f"Optimal threshold by F1: {best_thresh:.2f}, F1: {best_f1:.4f}")

    y_pred = (y_probs > best_thresh).astype(int)

    ## Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    print('Prediction with table data and radiomics features:')
    print(f'Accuracy: {acc * 100:.2f}%')
    print(f'AUC: {auc * 100:.2f}%')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[new_label_to_name[item] for item in set(y_test)]))

    return model, ids_train, ids_test, acc, auc

## @brief The main function for performing experiments and printing out cross validation results
def main():
    ## TODO: patient-wise accuracy
    ## TODO: optimal feature number
    
    ## Clinical data
    df_red = pd.read_csv("./R0_Patients_16.csv")
    df_lab = pd.read_csv("./Labdata_66_180_31.csv")
    
    df = pd.merge(df_red, df_lab, how="left", on="QHCCID")

    ## MR images
    df_1 = pd.read_csv("./radiomics_features.csv", index_col=0)
    df_1 = df_1.reset_index()
    df_1.rename(columns={'index': 'QHCCID'}, inplace=True)
    df_1["QHCCID"] = df_1["QHCCID"].apply(lambda x: x.replace("_1", ""))
    df_1.shape

    ## CT images
    df_2 = pd.read_csv("./radiomics_features2.csv", index_col=0)
    df_2 = df_2.reset_index()
    df_2.rename(columns={'index': 'QHCCID'}, inplace=True)
    df_2["QHCCID"] = df_2["QHCCID"].apply(lambda x: x.replace("_1", ""))
    df_2.shape

    ## Check the number of unique values in each column
    unique_counts_1 = df_1.nunique()
    unique_counts_2 = df_2.nunique()

    ## Filter out columns with only one unique value
    single_value_columns_1 = unique_counts_1[unique_counts_1 == 1].index
    single_value_columns_2 = unique_counts_2[unique_counts_2 == 1].index

    ## Drop these columns
    df_1_cleaned = df_1.drop(columns=single_value_columns_1)
    df_2_cleaned = df_2.drop(columns=single_value_columns_2)

    ## Rename columns
    df_1_cleaned.columns = ['QHCCID'] + [item + "_MR" for item in list(df_1_cleaned)[1:]]
    df_2_cleaned.columns = ['QHCCID'] + [item + "_CT" for item in list(df_2_cleaned)[1:]]

    ## Remove non-numeric columns
    df_1_cleaned = df_1_cleaned.drop(list(df_1_cleaned.select_dtypes(include=['object']).columns)[1:], axis=1)
    df_2_cleaned = df_2_cleaned.drop(list(df_2_cleaned.select_dtypes(include=['object']).columns)[1:], axis=1)

    ## Merge data
    new_df = df.copy()
    new_df = pd.merge(new_df, df_1_cleaned, how="left", on="QHCCID")
    new_df = pd.merge(new_df, df_2_cleaned, how="left", on="QHCCID")

    n_tabular = 47
    n_redcap = 16
    n_lab = 31
    n_mr = df_1_cleaned.shape[1] - 1
    n_ct = df_2_cleaned.shape[1] - 1
    start_mr = n_tabular
    start_ct = n_tabular + n_mr
    n_total = new_df.shape[1]

    combinations = {}

    def add_comb(name, df_slice):
        df_slice = df_slice.copy()
        df_slice[["QHCCID", "Recurrence"]] = new_df[["QHCCID", "Recurrence"]]
        combinations[name] = df_slice
        
    ## The different settings on using tabular data, CT, or MRI radiomics features
    add_comb("CT_MR", new_df.iloc[:, start_mr:])
    add_comb("CT", new_df.iloc[:, start_ct:])
    add_comb("MR", new_df.iloc[:, start_mr:start_ct])
    add_comb("Tab_MR", new_df.iloc[:, :start_ct])
    add_comb("Tab_CT", new_df.iloc[:, np.r_[0:n_tabular, start_ct:n_total]])
    add_comb("Red_CT", new_df.iloc[:, np.r_[0:n_redcap, start_ct:n_total]])
    add_comb("Red_MR", new_df.iloc[:, np.r_[0:n_redcap, start_mr:start_ct]])
    add_comb("Red_CT_MR", new_df.iloc[:, np.r_[0:n_redcap, start_mr:n_total]])
    add_comb("Lab_CT_MR", new_df.iloc[:, n_redcap:])
    add_comb("Lab_CT", new_df.iloc[:, np.r_[n_redcap:n_tabular, start_ct:n_total]])
    add_comb("Lab_MR", new_df.iloc[:, n_redcap:start_ct])
    add_comb("Red_Lab_CT_MR", new_df.iloc[:, np.r_[0:47, start_mr:n_total]])
    add_comb("Red", new_df.iloc[:, 0:16])
    add_comb("Lab", new_df.iloc[:, 16:47])
    add_comb("Red_Lab", new_df.iloc[:, 0:47])

    ## Perform patient-level random permutation cross validation
    ids_df = df["QHCCID"]
    y0 = df["Recurrence"]
    results = {}

    for name, df_comb in combinations.items():
        print(f"\n========== Running: {name} ==========")
        accs, aucs = [], []

        for xi in range(5):
            seed = 100 + xi
            np.random.seed(seed)

            ids_train, ids_test, _, _ = train_test_split(ids_df, y0, test_size=0.2, random_state=seed)
            _, _, _, acc, auc = experiment_redcap_lab_radiomics(df_comb, ids_train, ids_test)

            accs.append(acc)
            aucs.append(auc)
            print(f"  Fold {xi+1}: Accuracy = {acc:.2f}, AUC = {auc:.2f}")

        results[name] = {
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs)
        }

    ## Print overall results
    print("\n======== Cross-Validation Summary ========")
    for name, r in results.items():
        print(f"{name}: Accuracy = {r['acc_mean']:.2f} ± {r['acc_std']:.2f}, AUC = {r['auc_mean']:.2f} ± {r['auc_std']:.2f}")
    
if __name__ == "__main__":
    
    main()