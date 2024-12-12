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