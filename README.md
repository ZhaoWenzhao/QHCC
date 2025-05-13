# Baselines for machine-learning-based hepatocellular carcinoma diagnosis using multi-modal clinical data--QHCC

This repository provides machine learning baselines for two major tasks in hepatocellular carcinoma (HCC) diagnosis using the [QHCC dataset](#):  
1. **TNM staging prediction**
2. **Recurrence prediction**

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

## Extracting radiomics features
Download the data and modify the folder path in extract_QHCC_radiomics_features.py accordingly.

```bash
python extract_QHCC_radiomics_features.py
```

## Experiments on TNM staging

```bash
python demo_QHCC_XGBoost_comb.py
```

Cross-Validation Results on different combinations of input data:

| Data type |           Redcap data	        |	        Lab data	        |	    Redcap + Lab data	    |            Null	            |
|  :-------------: | :-------------: |  :-------------: |  :-------------: | :-------------: |
|	        |       ACC	    \|       AUC     |	    ACC     \|	    AUC	    |       ACC     \|	    AUC     |	    ACC     \|	    AUC     |
|    CT	    |   0.88 ± 0.04	\|   0.90 ± 0.06 |	0.51 ± 0.08	\|   0.68 ± 0.08	|   0.83 ± 0.07	\|   0.90 ± 0.02	|   0.55 ± 0.08	\|   0.64 ± 0.06 |
|    MRI    |	0.83 ± 0.12	\|   0.88 ± 0.09 |	0.49 ± 0.06	\|   0.65 ± 0.10	|   0.89 ± 0.05	\|   0.93 ± 0.03	|   0.51 ± 0.10	\|   0.57 ± 0.08 |
| CT + MRI	|   0.88 ± 0.05	\|   0.90 ± 0.07 |	0.55 ± 0.04	\|   0.69 ± 0.09	|   0.86 ± 0.03	\|   0.92 ± 0.02	|   0.54 ± 0.09	\|   0.60 ± 0.03 |
|   Null	|   0.82 ± 0.12	\|   0.88 ± 0.12 |	0.48 ± 0.06	\|   0.50 ± 0.06	|   0.85 ± 0.07	\|   0.90 ± 0.04	|   Null	    \|   Null        |

