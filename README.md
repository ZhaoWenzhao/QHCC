# QHCC
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
python demo_QHCC_XGBoost.py
```

Cross-Validation Results on prediction with Redcap data and lab data:
Mean Accuracy: 0.40 ± 0.07
Mean AUC: 0.45 ± 0.10

Cross-Validation Results on prediction with Redcap data, lab data, and radiomics features:
Mean Accuracy: 0.81 ± 0.08
Mean AUC: 0.86 ± 0.06