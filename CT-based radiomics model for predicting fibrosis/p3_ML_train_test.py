import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve, auc
import shap
from joblib import dump
random_state = 6

# === 1. Loading feature selection results and training data ===
# Load selected features information
feature_df = pd.read_excel("selected_features_with_coefficients.xlsx")
selected_features = feature_df['Feature'].tolist()

# Load training data with selected features
hstr_df = pd.read_csv('./ZS_V_Hfib_43.csv')  # High STR proportion group data
lstr_df = pd.read_csv('./ZS_V_Lfib_43.csv')  # Low STR proportion group data

# Shuffle datasets
hstr_df = hstr_df.sample(frac=1, random_state=random_state)
lstr_df = lstr_df.sample(frac=1, random_state=random_state)

# Combine test datasets
data_test = pd.concat([hstr_df, lstr_df], axis=0)

X_train = data_test[selected_features]
y_train = data_test['label']

standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
dump(standardscaler, 'model/standard_scaler.joblib')

X_train = pd.DataFrame(X_train)
print(f"Training data shape: {X_train.shape}, {y_train.shape}")

# === 2. model training and cross validation ===
# Initialize SVM model
model_svm = SVC(random_state=random_state, probability=True)

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1, 10],  # Kernel coefficient
    'kernel': ['linear', 'rbf']  # Kernel type
}

# Setup plot settings for ROC curve
plt.rcParams.update({
    'axes.labelsize': 10, 'axes.titlesize': 12,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8
})

# Setup JAMA color palette
JAMA_COLORS = ['#61B673', '#C5E0AB', '#C2C5A4', '#E5E7A2', '#E6D3A0']

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

# Storage for results
results = {
    'accuracy': [], 'auc': [], 'f1': [],
    'sensitivity': [], 'specificity': [],
    'fprs': [], 'tprs': [], 'roc_aucs': []
}

# For plotting mean ROC curve
mean_fpr = np.linspace(0, 1, 100)

# Perform 5-fold cross-validation
fig, ax = plt.subplots(figsize=(5, 5))

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n=== Processing Fold {fold_idx + 1} ===")

    # Split data for this fold
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        model_svm, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_fold_train, y_fold_train)

    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Validate model
    val_acc = best_model.score(X_fold_val, y_fold_val)
    results['accuracy'].append(val_acc)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Calculate probabilities and AUC
    y_val_proba = best_model.predict_proba(X_fold_val)[:, 1]
    val_auc = roc_auc_score(y_fold_val, y_val_proba)
    results['auc'].append(val_auc)
    print(f"Validation AUC: {val_auc:.4f}")

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_fold_val, y_val_proba)
    results['fprs'].append(fpr)
    results['tprs'].append(tpr)

    # Interpolate ROC curve to common FPR points
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    results['roc_aucs'].append(auc(mean_fpr, interp_tpr))

    # Plot fold ROC curve
    ax.plot(fpr, tpr, color=JAMA_COLORS[fold_idx], lw=1, alpha=0.4,
            label=f'Fold {fold_idx + 1} AUC={results["roc_aucs"][-1]:.2f}')

    # Calculate F1 score
    y_val_pred = best_model.predict(X_fold_val)
    f1 = f1_score(y_fold_val, y_val_pred, average='weighted')
    results['f1'].append(f1)
    print(f"F1 score: {f1:.4f}")

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_fold_val, y_val_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    results['sensitivity'].append(sensitivity)
    results['specificity'].append(specificity)
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

    # Save model
    # joblib.dump(best_model, f'model_svm_fold_{fold_idx + 1}.model')


# Calculate and display average metrics
avg_acc = np.mean(results['accuracy'])
avg_auc = np.mean(results['auc'])
avg_f1 = np.mean(results['f1'])
avg_sens = np.mean(results['sensitivity'])
avg_spec = np.mean(results['specificity'])

print("\n===== Cross-Validation Results =====")
print(f"Average Accuracy: {avg_acc:.4f}")
print(f"Average AUC: {avg_auc:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Average Sensitivity: {avg_sens:.4f}")
print(f"Average Specificity: {avg_spec:.4f}")

# Plot mean ROC curve
mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(results['fprs'], results['tprs'])], axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(results['roc_aucs'])

ax.plot(mean_fpr, mean_tpr, color='#415C42',
        label=f'Mean AUC={mean_auc:.2f} (±{std_auc:.2f})',
        lw=2, linestyle='-')

# Plot diagonal and finalize plot
ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='#6A6A6A')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Cross-Validation ROC Curves')
ax.legend(loc='lower right')
ax.grid(False)

plt.savefig('cv_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# === 3. Test on external dataset ===
# Load test data
try:
    hgg_test = pd.read_csv('./xy_V_Hfib_43.csv')
    lgg_test = pd.read_csv('./xy_V_Lfib_43.csv')
except:
    print("Test files not found, skipping external testing")
    exit()

# Combine test datasets
data_test = pd.concat([hgg_test, lgg_test], axis=0)

# Prepare test data using selected features
x_test_data = data_test[selected_features]
# x_test_data = x_test_data.astype(np.float32)

x_test_data = standardscaler.transform(x_test_data) #均值-标准差归一化
x_test_data = pd.DataFrame(x_test_data)

# x_test_data = standardscaler.transform(x_test_data) #均值-标准差归一化
# Prepare test labels
y_test_data = data_test['label']
test_results = []



# Set up ROC plot for test results
fig, ax = plt.subplots(figsize=(5, 5))
plt.rcParams.update({
    'axes.labelsize': 10, 'axes.titlesize': 12,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8
})

print(f"\nTesting on {len(x_test_data)} samples: HGG={len(hgg_test)}, LGG={len(lgg_test)}")

# Evaluate each fold's model on test set
for i in range(5):
    print(f"\n=== Testing with model from fold {i + 1} ===")

    try:
        # Load model and scaler
        model = joblib.load(f'model_svm_fold_{i + 1}.model')

        # Evaluate model
        acc = model.score(x_test_data, y_test_data)
        y_proba = model.predict_proba(x_test_data)[:, 1]
        auc_score = roc_auc_score(y_test_data, y_proba)
        y_pred = model.predict(x_test_data)
        f1 = f1_score(y_test_data, y_pred, average='weighted')

        # Compute confusion matrix
        cm = confusion_matrix(y_test_data, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # Store results
        fold_result = {
            'acc': acc, 'auc': auc_score, 'f1': f1,
            'sensitivity': sensitivity, 'specificity': specificity
        }
        test_results.append(fold_result)

        # Print results
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test AUC: {auc_score:.4f}")
        print(f"Test F1 score: {f1:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        print("Confusion matrix:")
        print(cm)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test_data, y_proba)
        ax.plot(fpr, tpr, color=JAMA_COLORS[i],
                lw=1 if i > 0 else 2, alpha=0.7,
                label=f'Fold {i + 1} AUC={auc_score:.2f}')

    except Exception as e:
        print(f"Error testing fold {i + 1}: {str(e)}")
        continue

# Calculate average test performance
if test_results:
    avg_acc = np.mean([r['acc'] for r in test_results])
    avg_auc = np.mean([r['auc'] for r in test_results])
    avg_f1 = np.mean([r['f1'] for r in test_results])
    avg_sens = np.mean([r['sensitivity'] for r in test_results])
    avg_spec = np.mean([r['specificity'] for r in test_results])

    print("\n===== Average Test Results =====")
    print(f"Accuracy: {avg_acc:.4f}")
    print(f"AUC: {avg_auc:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"Sensitivity: {avg_sens:.4f}")
    print(f"Specificity: {avg_spec:.4f}")

# Finalize test ROC plot
ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='#6A6A6A')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Test Set ROC Curves')
ax.legend(loc='lower right')
ax.grid(False)

plt.savefig('test_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()