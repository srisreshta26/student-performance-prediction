
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 1. LOAD THE DATASET
# ==================================================

data_path = Path("data/student-mat.csv")

if not data_path.exists():
    raise FileNotFoundError(
        "Dataset not found! Make sure student-mat.csv "
        "is inside the data folder."
    )

df = pd.read_csv(data_path, sep=";")

print("\n" + "=" * 55)
print("       STUDENT PERFORMANCE PREDICTION")
print("=" * 55)

print("\nDataset loaded successfully!")
print(f"Number of students: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nFirst five students:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nFinal grade (G3) summary:")
print(df["G3"].describe())


# ==================================================
# 2. EXPLORE THE FINAL GRADES
# ==================================================

plt.figure(figsize=(8, 5))

plt.hist(df["G3"], bins=10, edgecolor="black")

plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("grade_distribution.png")
plt.show()


# ==================================================
# 3. PREPARE INPUTS AND TARGET
# ==================================================

# The model will predict G3 (final grade).
# G3 must never be included as an input.

y = df["G3"]

# Model A: Predict BEFORE the first two exams.
# Exclude G1 and G2.
X_before = df.drop(columns=["G1", "G2", "G3"])

# Model B: Predict AFTER the first two exams.
# Include G1 and G2, but exclude G3.
X_after = df.drop(columns=["G3"])

# Use the same train/test rows for a fair comparison.
X_before_train, X_before_test, y_train, y_test = train_test_split(
    X_before,
    y,
    test_size=0.20,
    random_state=42
)

# Keep Model B aligned with the exact same students.
X_after_train = X_after.loc[X_before_train.index]
X_after_test = X_after.loc[X_before_test.index]


# ==================================================
# 4. BUILD A REUSABLE MACHINE LEARNING PIPELINE
# ==================================================

def build_model(X_train):
    """Create and train a model that handles numeric and text data."""

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Prepare numerical columns.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Prepare categorical columns.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Apply the correct preparation to each column type.
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Combine preparation and the prediction algorithm.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2
        ))
    ])

    model.fit(X_train, y_train)

    return model


# ==================================================
# 5. TRAIN BOTH MODELS
# ==================================================

print("\n" + "=" * 55)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 55)

print("\nTraining Model A: Before exams...")
model_before = build_model(X_before_train)

print("Training Model B: After the first two exams...")
model_after = build_model(X_after_train)

print("\nBoth models trained successfully!")


# ==================================================
# 6. EVALUATE BOTH MODELS
# ==================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate predictions using standard regression metrics."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n--- {model_name} ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.3f}")

    return predictions, {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


pred_before, results_before = evaluate_model(
    model_before,
    X_before_test,
    y_test,
    "Model A: Before Exams"
)

pred_after, results_after = evaluate_model(
    model_after,
    X_after_test,
    y_test,
    "Model B: After First Two Exams"
)


# ==================================================
# 7. COMPARE MODEL PERFORMANCE
# ==================================================

results_df = pd.DataFrame([
    results_before,
    results_after
])

print("\n" + "=" * 55)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 55)

print(results_df.round(3).to_string(index=False))

best_model = results_df.loc[results_df["MAE"].idxmin(), "Model"]

print(f"\nLower-MAE model on this test split: {best_model}")

results_df.to_csv("model_comparison.csv", index=False)


# ==================================================
# 8. SHOW EXAMPLE PREDICTIONS
# ==================================================

examples = pd.DataFrame({
    "Actual Final Grade": y_test.to_numpy(),
    "Predicted Before Exams": pred_before.round(1),
    "Predicted After Exams": pred_after.round(1)
})

print("\nExample predictions (first 10 test students):")
print(examples.head(10).to_string(index=False))

examples.to_csv("example_predictions.csv", index=False)


# ==================================================
# 9. VISUALIZE THE MODEL COMPARISON
# ==================================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["Before Exams", "After First Two Exams"],
    [results_before["MAE"], results_after["MAE"]],
    edgecolor="black"
)

plt.title("Prediction Error: Comparing Both Models")
plt.ylabel("Mean Absolute Error (lower is better)")

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


# ==================================================
# 10. FINISHED
# ==================================================

print("\n" + "=" * 55)
print("PROJECT RUN COMPLETED SUCCESSFULLY!")
print("=" * 55)

print("\nGenerated files:")
print("- grade_distribution.png")
print("- model_comparison.png")
print("- model_comparison.csv")
print("- example_predictions.csv")
