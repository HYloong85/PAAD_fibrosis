#### p1_extract_feature.py

We use **Pyradiomics** to extract the shape features, texture features and so on of CT images.

#### p2_select_feature.py

This script uses **T-test** and **Lasso** for two-stage feature selection.

#### p3_ML_train_test.py

This script is based on machine learning to train and test fibrosis prediction model.

#### p4_deploy_classifier.py

This script is a demo for deploying a classifier to classify the degree of fibrosis.

**data** is used to store the demo data for classifier inference.

**model** is used to store the weight of pre-trained fibrosis prediction model

**Result** is used to store the table of fibrosis prediction results, including the confidence of prediction and prediction results

**selected_features_with_coefficients.xlsx** is used to record the features selected by p2 and their coefficients. And it is used in the feature selection stage before model inference.