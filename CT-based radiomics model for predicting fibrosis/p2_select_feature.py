import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import levene, ttest_ind

random_seed = 6
np.random.seed(random_seed)

# === 1. Data Loading and Preparation ===
# Load STR proportion datasets
hstr_df = pd.read_csv('./train_Hstr.csv')  # High STR proportion group data
lstr_df = pd.read_csv('./train_Lstr.csv')  # Low STR proportion group data

# Shuffle datasets
hstr_df = hstr_df.sample(frac=1, random_state=random_seed)
lstr_df = lstr_df.sample(frac=1, random_state=random_seed)

# Remove non-numeric columns
def remove_non_numeric(df):
    """Drop columns containing string values"""
    str_cols = [col for col in df.columns
                if isinstance(df[col].iloc[0], str)]
    return df.drop(columns=str_cols)

hstr_df = remove_non_numeric(hstr_df)
lstr_df = remove_non_numeric(lstr_df)

# === 3. Feature Selection using T-tests ===
significant_features = []
# Iterate through all features (excluding label)
for feature in hstr_df.columns[1:]:
    _, p_levene = levene(hstr_df[feature], lstr_df[feature])
    use_equal_var = p_levene > 0.05  # True if variances are homogeneous
    _, p_ttest = ttest_ind(hstr_df[feature], lstr_df[feature],
                           equal_var=use_equal_var)
    # Retain significant features (p < 0.05)
    if p_ttest < 0.05:
        significant_features.append(feature)
print(f"{len(significant_features)} significant features after t-tests")

# Create filtered dataset
selected_cols = ['label'] + significant_features
filtered_data = pd.concat([
    hstr_df[selected_cols],
    lstr_df[selected_cols]
]).sample(frac=1, random_state=random_seed)  # Shuffle

# Separate features and labels
X = filtered_data[significant_features]
y = filtered_data['label']

# === 4. Lasso Feature Selection ===
# Standardize features (critical for Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=significant_features)

alphas = np.logspace(-3, -2, 50, base=5)

# Initialize LassoCV model
lasso_model = LassoCV(
    alphas=alphas,
    cv=5,
    max_iter=1000,
    random_state=random_seed
)

# Train model using standardized features
lasso_model.fit(X_scaled, y)

# Extract coefficients and identify selected features
coef_series = pd.Series(lasso_model.coef_, index=significant_features)
selected_features = coef_series[coef_series != 0].index

print(f"\nLasso selected {len(selected_features)} features:")
print(selected_features.tolist())

X_selected = X_scaled[selected_features]

# === 5. Save Selected Features for Future Use ===
# Create DataFrame with selected features and their coefficients
feature_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': coef_series[selected_features]
})

# Save feature list with coefficients
feature_df.to_excel("selected_features_with_coefficients.xlsx", index=False)

# Prepare the filtered dataset with only selected features
final_data = pd.concat([
    X_selected,
    y.reset_index(drop=True)  # Align indices by resetting
], axis=1)

# Save the filtered dataset with selected features
final_data.to_excel("data_with_selected_features.xlsx", index=False)

print("\nResults saved successfully:")
print(f" - Feature list: selected_features_with_coefficients.xlsx")
print(f" - Dataset: data_with_selected_features.xlsx")