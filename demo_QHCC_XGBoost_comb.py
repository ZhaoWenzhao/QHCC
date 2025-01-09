## @package demo_QHCC_XGBoost
# @version v1.1
# @brief The demo file for doing classification on QHCC datasets 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

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
    X_df = new_df.drop("TNM Stage", axis=1)
    y_df = new_df[["TNM Stage"]]

    ## Use label encoding to convert text labels to numeric data and fill missing values with 'unknown'
    label_encoder = LabelEncoder()
    for column in X_df.select_dtypes(include=['object']).columns:
        X_df[column] = label_encoder.fit_transform(X_df[column].fillna('unknown'))

    y_df.loc[:, 'TNM Stage'] = y_df['TNM Stage'].fillna('unknown')
    y_df.loc[:, 'label'] = label_encoder.fit_transform(y_df['TNM Stage'])

    ## Create a label-to-name dictionary for easy observation of results
    label_to_name = dict(zip(y_df.drop_duplicates()["label"], y_df.drop_duplicates()["TNM Stage"]))

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
    
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_class, 
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    ## Feature selection to select the features with the highest mutual information values
    X_train_imputed = X_imputed[[np.isin(i, ids_train).item() for i in ids_df ] ]
    
    min_features_to_select = X_train_imputed.shape[1]//2 # Minimum number of features to consider
    clf = model 
    cv = StratifiedKFold(5)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(X_train_imputed, y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")    
    num_features = rfecv.n_features_
    
    selected_features = auto_feature_select(X_train_imputed, y_train, k=num_features)
    ## Retain the top 15 features with the highest mutual information values
    X_train = X_train.iloc[:, selected_features]
    X_test = X_test.iloc[:, selected_features]


    ## Train XGBoost model 
    ## Training
    model.fit(X_train, y_train)

    ## Make prediction
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    ## Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score( y_test,  y_pred_proba,  multi_class='ovr')

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
    new_X_df = new_df.drop("TNM Stage", axis=1)
    new_y_df = new_df[["TNM Stage"]]

    label_encoder = LabelEncoder()
    for column in new_X_df.select_dtypes(include=['object']).columns:
        new_X_df[column] = label_encoder.fit_transform(new_X_df[column].fillna('unknown'))

    new_y_df.loc[:, 'TNM Stage'] = new_y_df['TNM Stage'].fillna('unknown')
    new_y_df.loc[:, 'label'] = label_encoder.fit_transform(new_y_df['TNM Stage'])

    new_label_to_name = dict(zip(new_y_df.drop_duplicates()["label"], new_y_df.drop_duplicates()["TNM Stage"]))

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

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_class, 
        use_label_encoder=False,
        eval_metric='mlogloss'
    )


    ## Feature selection to select the top k features with the highest mutual information values
    new_X_train_imputed = new_X_imputed[[np.isin(i, ids_train).item() for i in id_new_df ] ]
    #breakpoint()
    ##############################################################
    min_features_to_select = new_X_train_imputed.shape[1]//2 #  # Minimum number of features to consider
    clf = model #LogisticRegression()#
    cv = StratifiedKFold(5)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(new_X_train_imputed, y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")    
    num_features = rfecv.n_features_
    ###############################################################
    
    selected_features = auto_feature_select(new_X_train_imputed, y_train, k=num_features)
    ## Retain the top k features with the highest mutual information values
    X_train = X_train.iloc[:, selected_features]
    X_test = X_test.iloc[:, selected_features]

    ## Train XGBoost model 
    ## Training
    model.fit(X_train, y_train)

    ## Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    ## Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score( y_test,  y_pred_proba,  multi_class='ovr')
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
    df_red = pd.read_csv("./data/redcap_data_m.csv")
    df_lab = pd.read_csv("./data/lab_data_m.csv")
    
    df = pd.merge(df_red, df_lab, how="left", on="QHCCID")

    ## Preprocess labels for a balanced distribution
    df["TNM Stage"] = df["TNM Stage"].apply(lambda x: x[:2] if isinstance(x, str) else np.nan)
    df = df.dropna(how='all', subset=["TNM Stage"])
    df["TNM Stage"] = df["TNM Stage"].apply(lambda x: "TX, T0 or T1" if x=="T0" or x=="T1" or x=="TX"  else x)
    df["TNM Stage"] = df["TNM Stage"].apply(lambda x: "T3 or T4" if x=="T3" or x=="T4" else x)

    ## MR images
    df_1 = pd.read_csv("./data/radiomics_features.csv", index_col=0)
    df_1 = df_1.reset_index()
    df_1.rename(columns={'index': 'QHCCID'}, inplace=True)
    df_1["QHCCID"] = df_1["QHCCID"].apply(lambda x: x.replace("_1", ""))
    df_1.shape

    ## CT images
    df_2 = pd.read_csv("./data/radiomics_features2.csv", index_col=0)
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

    new_df_Tab = new_df.iloc[:,:86]
    
    new_df_Lab = new_df.iloc[:,32:86]# lab data only
    new_df_Lab[["QHCCID","TNM Stage"]] = new_df[["QHCCID","TNM Stage"]]####

    ids_df = df["QHCCID"]
    y0 = df["TNM Stage"]
    
    ## The different settings on using tabular data, CT, or MRI radiomics features
    ## Radiomics features only
    new_df_CT_MR = new_df.iloc[:,86:]#
    new_df_CT_MR["QHCCID"] = new_df["QHCCID"]
    new_df_CT_MR["TNM Stage"] = new_df["TNM Stage"]
    ## Radiomics feautures from CT only
    new_df_CT = new_df.iloc[:,198:]#
    new_df_CT["QHCCID"] = new_df["QHCCID"]
    new_df_CT["TNM Stage"] = new_df["TNM Stage"]
    ## Radiomics feautures from MR only
    new_df_MR = new_df.iloc[:,86:198]#
    new_df_MR["QHCCID"] = new_df["QHCCID"]
    new_df_MR["TNM Stage"] = new_df["TNM Stage"]
    ## Tabular data + radiomics feautures from MR
    new_df_Tab_MR = new_df.iloc[:,:198]#
    new_df_Tab_MR["QHCCID"] = new_df["QHCCID"]
    new_df_Tab_MR["TNM Stage"] = new_df["TNM Stage"]
    ## Tabular data + radiomics feautures from CT
    ind = np.array(np.arange(86).tolist() + [ i+198 for i in np.arange(309-198)])
    new_df_Tab_CT = new_df.iloc[:,ind]#
    new_df_Tab_CT["QHCCID"] = new_df["QHCCID"]
    new_df_Tab_CT["TNM Stage"] = new_df["TNM Stage"]
    ## Redcap data + radiomics feautures from CT
    ind = np.array(np.arange(32).tolist() + [ i+198 for i in np.arange(309-198)])
    new_df_Red_CT = new_df.iloc[:,ind]#
    new_df_Red_CT["QHCCID"] = new_df["QHCCID"]
    new_df_Red_CT["TNM Stage"] = new_df["TNM Stage"]
    ## Redcap data + radiomics feautures from MR
    ind = np.array(np.arange(32).tolist() + [ i+86 for i in np.arange(198-86)])
    new_df_Red_MR = new_df.iloc[:,ind]#
    new_df_Red_MR["QHCCID"] = new_df["QHCCID"]
    new_df_Red_MR["TNM Stage"] = new_df["TNM Stage"]
    ## Redcap data + radiomics feautures from CT and MR
    ind = np.array(np.arange(32).tolist() + [ i+86 for i in np.arange(309-86)])
    new_df_Red_CT_MR = new_df.iloc[:,ind]#
    new_df_Red_CT_MR["QHCCID"] = new_df["QHCCID"]
    new_df_Red_CT_MR["TNM Stage"] = new_df["TNM Stage"]
    ## Lab data + radiomics feautures from CT and MR
    new_df_Lab_CT_MR = new_df.iloc[:,32:]#
    new_df_Lab_CT_MR["QHCCID"] = new_df["QHCCID"]
    new_df_Lab_CT_MR["TNM Stage"] = new_df["TNM Stage"]
    ## Lab data + radiomics feautures from CT
    ind = np.array([ i+32 for i in np.arange(86-32)] + [ i+198 for i in np.arange(309-198)])
    new_df_Lab_CT = new_df.iloc[:,ind]#
    new_df_Lab_CT["QHCCID"] = new_df["QHCCID"]
    new_df_Lab_CT["TNM Stage"] = new_df["TNM Stage"]
    ## Lab data + radiomics feautures from MR    
    new_df_Lab_MR = new_df.iloc[:,32:198]#
    new_df_Lab_MR["QHCCID"] = new_df["QHCCID"]
    new_df_Lab_MR["TNM Stage"] = new_df["TNM Stage"]
    
    ## Metrics to track
    accuracies0 = []
    aucs0 = []
    accuracies1 = []
    aucs1 = []
    
    ## Perform patient-level random permutation cross validation
    for xi in np.arange(5):
        
        seed = 100 + xi
        
        np.random.seed(seed)
        
        ids_train, ids_test, _, __ = train_test_split(ids_df, y0, test_size=0.20, random_state=seed)
        
        print(f"Round {xi+1}") 

        _, ids_train, ids_test, acc0, auc0 = experiment_redcap_lab(new_df_Tab, ids_train, ids_test, seed = seed)   
        accuracies0.append(acc0)
        aucs0.append(auc0)
        
        _, ids_train, ids_test, acc1, auc1 = experiment_redcap_lab_radiomics(new_df, ids_train, ids_test)
        accuracies1.append(acc1)
        aucs1.append(auc1)
    
    ## Print overall results
    print("\nCross-Validation Results on prediction with Redcap data and lab data:")
    print(f"Mean Accuracy: {np.mean(accuracies0):.2f} ± {np.std(accuracies0):.2f}")
    print(f"Mean AUC: {np.mean(aucs0):.2f} ± {np.std(aucs0):.2f}")
    print("\nCross-Validation Results on prediction with Redcap data, lab data, and radiomics features:")
    print(f"Mean Accuracy: {np.mean(accuracies1):.2f} ± {np.std(accuracies1):.2f}")
    print(f"Mean AUC: {np.mean(aucs1):.2f} ± {np.std(aucs1):.2f}")
    
if __name__ == "__main__":
    
    main()
