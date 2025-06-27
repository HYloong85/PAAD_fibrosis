import numpy as np
import pandas as pd
import joblib

# 1. Load test dataset
test_data = pd.read_excel('./data/data_demo.xlsx')
X_test = test_data.values
x_test_data = test_data[test_data.columns[1:]]

# Select features
feature_df = pd.read_excel("selected_features_with_coefficients.xlsx")
selected_features = feature_df['Feature'].tolist()
x_test_data = x_test_data[selected_features]

columnNames = x_test_data.columns
x_test_data = x_test_data.astype(np.float32)

# Standardize the test data
standardscaler = joblib.load('./model/standard_scaler.joblib')  # 保存的标准化器
x_test_data = standardscaler.fit_transform(x_test_data) #均值-标准差归一化
x_test_data = pd.DataFrame(x_test_data,columns=columnNames)

# 3. Load pre-trained SVM model
model = joblib.load('./model/model_svm_fold_1.model')

# 4. Make predictions
y_pred = model.predict(x_test_data)
y_prob = model.predict_proba(x_test_data)  # 获取预测概率

# 5. Create results DataFrame
# Add meaningful class labels
class_map = {0: "Low Fibrosis", 1: "High Fibrosis"}
pred_labels = [class_map[pred] for pred in y_pred]

# Create complete results table
results = pd.DataFrame({
    'Patient_ID': test_data[test_data.columns[0]],
    'Prediction': pred_labels,
    'Confidence': np.max(y_prob, axis=1).round(4) * 100,  # Confidence percentage
    'Low_Fibrosis_Probability': y_prob[:, 0].round(4),
    'High_Fibrosis_Probability': y_prob[:, 1].round(4)
})

# 6. Display individual patient results
print("\n" + "=" * 50)
print("INDIVIDUAL PATIENT RESULTS")
print("=" * 50)

# Print detailed results for each patient
for i, row in results.iterrows():
    print(f"\n► Patient ID: {row['Patient_ID']}")
    print(f"  Prediction: {row['Prediction']}")
    print(f"  Confidence: {row['Confidence']}%")
    print(f"  Low Fibrosis Probability: {row['Low_Fibrosis_Probability'] * 100:.1f}%")
    print(f"  High Fibrosis Probability: {row['High_Fibrosis_Probability'] * 100:.1f}%")

    # Visual confidence indicator
    conf_level = min(100, max(0, int(row['Confidence'])))
    print(f"  Confidence: [{'■' * (conf_level // 5)}{'□' * (20 - conf_level // 5)}] {conf_level}%")


# 7. Save results to CSV
save_name = './result/RESULTS.csv'
results.to_csv(save_name, index=False)
print("\n" + "=" * 50)
print(f"Prediction results saved to  {save_name}")


