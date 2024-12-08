
Tú
27/11/2024
grupo_Caled_GuidovanRossum.zip
ZIP•357 kB
grupo_Caled_GuidovanRossum.zip
17:48
/Descargas/Gramaticas_Libro/Entrega procesadores/pl_yacc/java/src/caled. Lo del jflex y CUP
17:50
/Escritorio/ProcesadoresLenguaje/java/src/parser
17:51
Elegir carreteras principales de uk
18:43
Hacerlo de eso
18:43
28/11/2024
try{

			new parser(new Yylex(reader)).parse();
		}
		catch ( Exception e) {
			System.out.println("¡¡ El análisis es INCORRECTO!!");
			System.exit(1);
		}
		System.out.println("¡¡ El Análisis es Correcto !!");
	
		}
:}
19:35
LUNES
Notebooks.zip
ZIP•65 MB
Notebooks.zip
16:06
MAHINE LEARNING.zip
ZIP•79 MB
MAHINE LEARNING.zip
16:48
pip install google-cloud-speech google-cloud-translate google-cloud-texttospeech
23:20
MARTES
Añadir más errores en el CUP
23:05
JUEVES
HIPOTESIS_CONCLUSIONES.zip
ZIP•6 MB
20:36
En este caso, es curioso los resultados que nos ha arrojado el DBSCAN puesto que, como se puede observar, son muy similares los clusterers que nos genera con respecto a los anteriores pero los nombres mostrados están invertidos aqui. Asimismo, se han hecho diversas iteraciones para encontrar los valores con los cuales se obtiene mejor coeficente de Silhouette en donde se han ido variando los valores correspondientes al epsilon, la cual es la distancia máxima para considerar puntos como vecinos, y el número mínimo
20:51
random_forest.py
PY•4 kB
23:32
MAHINE LEARNING.zip
ZIP•79 MB
23:39
AYER
twitter.com
https://twitter.com/SJBpuertollano/status/1857821418190242300
twitter.com
https://twitter.com/SJBpuertollano/status/1857821418190242300
15:50
HOY
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
training_set_features = pd.read_csv('training_set_features.csv')
training_set_labels = pd.read_csv('training_set_labels.csv')
test_set_features = pd.read_csv('test_set_features.csv')

# Merge training features and labels
training_data = pd.merge(training_set_features, training_set_labels, on="respondent_id")

# Drop respondent_id from features (not useful for prediction)
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

# Split data into training and validation sets
X_train, X_val, y_h1n1_train, y_h1n1_val = train_test_split(X, y_h1n1, test_size=0.3, random_state=42)
X_train_seasonal, X_val_seasonal, y_seasonal_train, y_seasonal_val = train_test_split(X, y_seasonal, test_size=0.3, random_state=42)

# Train Random Forest for h1n1_vaccine
rf_h1n1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_h1n1.fit(X_train, y_h1n1_train)

# Train Random Forest for seasonal_vaccine
rf_seasonal = RandomForestClassifier(n_estimators=100, random_state=42)
rf_seasonal.fit(X_train_seasonal, y_seasonal_train)

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
print("Submission file saved as 'submission.csv'.")
22:51



