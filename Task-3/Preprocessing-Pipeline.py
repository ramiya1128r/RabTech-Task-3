
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Credit Card Default Dataset
credit_data = fetch_ucirepo(id=350)

X = credit_data.data.features
y = credit_data.data.targets.squeeze()

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)

# Step 2: Identify categorical and numerical features
categorical_features = ["X2", "X3", "X4"]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# Step 3: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Step 4: Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# Step 5: Create Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate model
print("\nMODEL ACCURACY:")
print(accuracy_score(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nPREPROCESSING PIPELINE COMPLETED!")
