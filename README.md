# Baselines for machine-learning-based hepatocellular carcinoma diagnosis using multi-modal clinical data--QHCC

This repository provides machine learning baselines for two major tasks in hepatocellular carcinoma (HCC) diagnosis using the QHCC dataset:  
1. **TNM staging classification**

- XGBoost-based model

2. **Recurrence prediction**

- XGBoost-based model
- TabPFN-based model
  
Both methods use the same multi-modal feature inputs and follow similar data preprocessing steps.

All experiments are based on multi-modal data: clinical data, laboratory data, and imaging radiomics (CT/MRI).

## The example codes about how to use the QHCC dataset for machine learning tasks

## Dependencies

* Python-3.8.8
* numpy-1.21.5
* pandas-1.4.2
* sklearn-1.1.2
* xgboost-2.1.3
* pyradiomics-3.0.1
* SimpleITK-2.4.0
* imbalanced-learn-0.10.1
* pymfe-0.4.3
* tabpfn-2.0.9

## Extracting radiomics features
Download the data and modify the folder path in extract_QHCC_radiomics_features.py accordingly.

```bash
python extract_QHCC_radiomics_features.py
```

## Experiments on Recurrence prediction using TabPFN

```bash
python demo_QHCC_Tabpfn_comb_recurrence.py
```

Cross-Validation Results on different combinations of input data:

| Modality     | Redcap (ACC &#124; AUC)   | Lab (ACC &#124; AUC)      | Redcap + Lab (ACC &#124; AUC) | Null (ACC &#124; AUC)        |
|--------------|----------------------------|----------------------------|-------------------------------|------------------------------|
|   CT         | 0.73 ± 0.08 &#124; 0.67 ± 0.10 | 0.81 ± 0.05 &#124; 0.77 ± 0.05 | 0.66 ± 0.08 &#124; 0.70 ± 0.07     | 0.69 ± 0.09 &#124; 0.68 ± 0.06 |
|   MRI        | 0.63 ± 0.08 &#124; 0.56 ± 0.10 | 0.76 ± 0.08 &#124; 0.77 ± 0.09 | 0.76 ± 0.13 &#124; 0.69 ± 0.33     | 0.61 ± 0.05 &#124; 0.55 ± 0.13 |
|   CT + MRI   | 0.74 ± 0.06 &#124; 0.67 ± 0.10 | 0.75 ± 0.08 &#124; 0.81 ± 0.07 | 0.82 ± 0.03 &#124; 0.84 ± 0.04     | 0.77 ± 0.05 &#124; 0.72 ± 0.07 |
|   Null       | 0.52 ± 0.23 &#124; 0.39 ± 0.23 | 0.71 ± 0.10 &#124; 0.82 ± 0.03 | 0.63 ± 0.23 &#124; 0.61 ± 0.29     | Null &#124; Null              |

