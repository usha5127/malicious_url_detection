# ================================
# Malicious URL Detection System
# ================================

# --------- Imports ---------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tldextract
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# --------- Load Dataset ---------
# Place dataset.csv in the same folder as this file
df = pd.read_csv("dataset.csv")
print("Dataset Loaded Successfully")
print(df.head())

# --------- Column Standardization ---------
df.rename(columns=lambda x: x.strip().lower(), inplace=True)
df.rename(columns={'url': 'url', 'label': 'label', 'class': 'label', 'type': 'label'}, inplace=True)

print("Columns after renaming:", df.columns)

# --------- Handle Missing Values ---------
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df.dropna(inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# --------- Label Distribution ---------
if 'label' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['label'])
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

# --------- Feature Extraction ---------
def extract_features(url):
    domain_info = tldextract.extract(url)
    return {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_underscores': url.count('_'),
        'num_slashes': url.count('/'),
        'num_digits': sum(c.isdigit() for c in url),
        'has_https': int('https' in url),
        'subdomain_length': len(domain_info.subdomain)
    }

if 'url' in df.columns:
    features_df = df['url'].apply(lambda x: pd.Series(extract_features(str(x))))
    df = pd.concat([df, features_df], axis=1)
    df.drop(columns=['url'], inplace=True)

print("\nFeature extraction completed")
print(df.head())

# --------- Encode Labels ---------
df['label'] = df['label'].astype(str)
df['label'] = pd.factorize(df['label'])[0]

# --------- Encode Categorical Columns ---------
non_numeric_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

print("\nDataset after encoding:")
print(df.head())

# --------- Train-Test Split ---------
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------- Model Training ---------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel training completed")

# --------- Evaluation ---------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------- Confusion Matrix ---------
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --------- Save Outputs ---------
df.to_csv("cleaned_urls.csv", index=False)
joblib.dump(model, "malicious_url_model.pkl")

feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

print("\nFiles saved:")
print(os.listdir())

# --------- Sample Prediction ---------
model = joblib.load("malicious_url_model.pkl")
trained_features = joblib.load("feature_columns.pkl")

sample_url_features = pd.DataFrame([{
    "url_length": 25,
    "num_dots": 2,
    "num_hyphens": 1,
    "num_underscores": 0,
    "num_slashes": 3,
    "num_digits": 1,
    "has_https": 1,
    "subdomain_length": 4
}])

sample_url_features = sample_url_features.reindex(
    columns=trained_features, fill_value=0
)

prediction = model.predict(sample_url_features)
print("\nPrediction Result:", "🔴 Malicious" if prediction[0] == 1 else "🟢 Safe")
