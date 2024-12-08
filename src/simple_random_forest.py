import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    accuracy_score, f1_score, recall_score, precision_score)


# Print roc function
def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_title(f"ROC Curve for {label_name}")
    ax.legend(loc="lower right")
    
def load_data():
	training_set_features = pd.read_csv('../Data/training_set_features.csv')
	training_set_labels = pd.read_csv('../Data/training_set_labels.csv')
	test_set_features = pd.read_csv('../Data/test_set_features.csv')
	return training_set_features, training_set_labels, test_set_features

def preprocess_data(training_set_features, training_set_labels, test_set_features):
	# Merge training features and labels
	training_data = pd.merge(training_set_features, training_set_labels, on="respondent_id")

	# Drop respondent_id from features 
	X = training_data.drop(columns=["respondent_id", "h1n1_vaccine", "seasonal_vaccine"])
	y_h1n1 = training_data["h1n1_vaccine"]
	y_seasonal = training_data["seasonal_vaccine"]
	test_X = test_set_features.drop(columns=["respondent_id"])

	# Handle missing values
	imputer = SimpleImputer(strategy="most_frequent")
	X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
	test_X = pd.DataFrame(imputer.transform(test_X), columns=test_X.columns)

	# Encode categorical variables
	categorical_cols = X.select_dtypes(include=["object"]).columns
	encoder = LabelEncoder()
	for col in categorical_cols:
		X[col] = encoder.fit_transform(X[col])
		test_X[col] = encoder.transform(test_X[col])

	return X, y_h1n1, y_seasonal, test_X

def train_model(X, y_h1n1, y_seasonal):
	# Split data into training and validation sets
	X_train, X_val, y_h1n1_train, y_h1n1_val = train_test_split(X, y_h1n1, test_size=0.3, random_state=42)
	X_train_seasonal, X_val_seasonal, y_seasonal_train, y_seasonal_val = train_test_split(X, y_seasonal, test_size=0.3, random_state=42)

	# Train Random Forest for h1n1_vaccine
	rf_h1n1 = RandomForestClassifier(n_estimators=100, random_state=42)
	rf_h1n1.fit(X_train, y_h1n1_train)

	# Train Random Forest for seasonal_vaccine
	rf_seasonal = RandomForestClassifier(n_estimators=100, random_state=42)
	rf_seasonal.fit(X_train_seasonal, y_seasonal_train)

	return rf_h1n1, rf_seasonal, X_val, y_h1n1_val, X_val_seasonal, y_seasonal_val

def evaluate_model(rf_h1n1, rf_seasonal, X_val, y_h1n1_val, X_val_seasonal, y_seasonal_val):
	# Predict probabilities for validation sets
	y_h1n1_val_probs = rf_h1n1.predict_proba(X_val)[:, 1]
	y_seasonal_val_probs = rf_seasonal.predict_proba(X_val_seasonal)[:, 1]

	# Create subplots for the ROC curves
	fig, ax = plt.subplots(1, 2, figsize=(12, 6))

	# Plot ROC for H1N1 Vaccine
	plot_roc(y_h1n1_val, y_h1n1_val_probs, 'H1N1 Vaccine', ax=ax[0])

	# Plot ROC for Seasonal Vaccine
	plot_roc(y_seasonal_val, y_seasonal_val_probs, 'Seasonal Vaccine', ax=ax[1])

	# Adjust layout and display the plot
	fig.tight_layout()
	plt.show()

	# Calculate average AUC for competition metric
	y_eval = np.column_stack((y_h1n1_val, y_seasonal_val))
	y_probs = np.column_stack((y_h1n1_val_probs, y_seasonal_val_probs))
	average_auc = roc_auc_score(y_eval, y_probs, average="macro")
	print(f"Average ROC AUC: {average_auc:.4f}")

def final_training_and_prediction(rf_h1n1, rf_seasonal, X, y_h1n1, y_seasonal, test_X):
	# Final training on the full training set
	rf_h1n1.fit(X, y_h1n1)
	rf_seasonal.fit(X, y_seasonal)

	# Predict on the test set
	test_set_features["h1n1_vaccine"] = rf_h1n1.predict_proba(test_X)[:, 1]
	test_set_features["seasonal_vaccine"] = rf_seasonal.predict_proba(test_X)[:, 1]

	# Save the predictions
	submission = test_set_features[["respondent_id"]]
	submission["h1n1_vaccine"] = test_set_features["h1n1_vaccine"]
	submission["seasonal_vaccine"] = test_set_features["seasonal_vaccine"]

	# Save submission file
	submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
		training_set_features, training_set_labels, test_set_features = load_data()
		X, y_h1n1, y_seasonal, test_X = preprocess_data(training_set_features, training_set_labels, test_set_features)
		rf_h1n1, rf_seasonal, X_val, y_h1n1_val, X_val_seasonal, y_seasonal_val = train_model(X, y_h1n1, y_seasonal)
		evaluate_model(rf_h1n1, rf_seasonal, X_val, y_h1n1_val, X_val_seasonal, y_seasonal_val)
		final_training_and_prediction(rf_h1n1, rf_seasonal, X, y_h1n1, y_seasonal, test_X)
