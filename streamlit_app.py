
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Explore how student information can help estimate final grades. "
    "This is an educational machine learning project, not a guarantee "
    "of any student's performance."
)


# ==========================================
# 2. LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    path = Path("data/student-mat.csv")

    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. Put student-mat.csv inside the data folder."
        )

    return pd.read_csv(path, sep=";")


@st.cache_resource
def train_models():
    data = load_data()

    y = data["G3"]
    X_before = data.drop(columns=["G1", "G2", "G3"])
    X_after = data.drop(columns=["G3"])

    X_before_train, X_before_test, y_train, y_test = train_test_split(
        X_before,
        y,
        test_size=0.20,
        random_state=42
    )

    X_after_train = X_after.loc[X_before_train.index]
    X_after_test = X_after.loc[X_before_test.index]

    def build_model(X_train):
        numeric_columns = X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = X_train.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ])

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

    before_model = build_model(X_before_train)
    after_model = build_model(X_after_train)

    before_predictions = before_model.predict(X_before_test)
    after_predictions = after_model.predict(X_after_test)

    metrics = pd.DataFrame([
        {
            "Model": "Before exams",
            "MAE": mean_absolute_error(y_test, before_predictions),
            "RMSE": mean_squared_error(
                y_test, before_predictions
            ) ** 0.5,
            "R²": r2_score(y_test, before_predictions)
        },
        {
            "Model": "After first two exams",
            "MAE": mean_absolute_error(y_test, after_predictions),
            "RMSE": mean_squared_error(
                y_test, after_predictions
            ) ** 0.5,
            "R²": r2_score(y_test, after_predictions)
        }
    ])

    return before_model, after_model, metrics, data


try:
    with st.spinner("Loading data and training the models..."):
        before_model, after_model, metrics, data = train_models()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.success("Both models are ready!")


# ==========================================
# 3. SHOW DATASET OVERVIEW
# ==========================================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Students", len(data))
col2.metric("Features", data.shape[1])
col3.metric("Missing values", int(data.isnull().sum().sum()))

with st.expander("Preview the student dataset"):
    st.dataframe(data.head(10), use_container_width=True)


# ==========================================
# 4. COMPARE MODEL PERFORMANCE
# ==========================================

st.header("📈 Compare the Two Models")

st.write(
    "MAE and RMSE measure prediction error, so lower is better. "
    "R² describes how well predictions explain grade variation."
)

st.dataframe(
    metrics.round(3),
    hide_index=True,
    use_container_width=True
)

st.bar_chart(metrics.set_index("Model")["MAE"])

st.caption(
    "These metrics come from one held-out test set. "
    "Results may differ on other students."
)


# ==========================================
# 5. PREDICTION MODE
# ==========================================

st.header("🔮 Predict a Final Grade")

mode = st.radio(
    "When do you want to estimate the student's final grade?",
    ["Before exams", "After first two exams"],
    horizontal=True
)

st.write("Enter the student's details below.")

# Values and categories are taken from the dataset.
with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox("School", sorted(data["school"].unique()))
        sex = st.selectbox("Sex", sorted(data["sex"].unique()))
        age = st.slider("Age", 15, 22, 16)
        address = st.selectbox("Home address type", sorted(data["address"].unique()))
        famsize = st.selectbox("Family size", sorted(data["famsize"].unique()))
        Pstatus = st.selectbox(
            "Parents' cohabitation status",
            sorted(data["Pstatus"].unique())
        )
        Medu = st.select_slider(
            "Mother's education level",
            options=[0, 1, 2, 3, 4],
            value=2
        )
        Fedu = st.select_slider(
            "Father's education level",
            options=[0, 1, 2, 3, 4],
            value=2
        )
        Mjob = st.selectbox("Mother's job", sorted(data["Mjob"].unique()))
        Fjob = st.selectbox("Father's job", sorted(data["Fjob"].unique()))
        reason = st.selectbox("Reason for choosing school", sorted(data["reason"].unique()))
        guardian = st.selectbox("Guardian", sorted(data["guardian"].unique()))
        traveltime = st.select_slider("Travel time", options=[1, 2, 3, 4], value=1)
        studytime = st.select_slider("Weekly study time", options=[1, 2, 3, 4], value=2)
        failures = st.select_slider("Past class failures", options=[0, 1, 2, 3], value=0)
        absences = st.number_input("School absences", min_value=0, max_value=100, value=2)

    with col2:
        schoolsup = st.selectbox("Extra educational support", ["yes", "no"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Extra paid classes", ["yes", "no"])
        activities = st.selectbox("Extracurricular activities", ["yes", "no"])
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet access at home", ["yes", "no"])
        romantic = st.selectbox("In a romantic relationship", ["yes", "no"])
        famrel = st.slider("Family relationship quality", 1, 5, 4)
        freetime = st.slider("Free time after school", 1, 5, 3)
        goout = st.slider("Going out with friends", 1, 5, 3)
        Dalc = st.slider("Workday alcohol consumption scale", 1, 5, 1)
        Walc = st.slider("Weekend alcohol consumption scale", 1, 5, 1)
        health = st.slider("Self-reported health", 1, 5, 3)

        G1 = None
        G2 = None

        if mode == "After first two exams":
            st.subheader("Earlier exam grades")
            G1 = st.number_input("First period grade (G1)", 0, 20, 10)
            G2 = st.number_input("Second period grade (G2)", 0, 20, 10)

    submitted = st.form_submit_button("Predict Final Grade")


# ==========================================
# 6. MAKE THE PREDICTION
# ==========================================

if submitted:
    student = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }

    if mode == "After first two exams":
        student["G1"] = G1
        student["G2"] = G2
        selected_model = after_model
    else:
        selected_model = before_model

    input_df = pd.DataFrame([student])

    prediction = float(selected_model.predict(input_df)[0])
    prediction = max(0.0, min(20.0, prediction))

    st.subheader("Prediction Result")
    st.metric("Estimated final grade", f"{prediction:.1f} / 20")

    st.progress(prediction / 20)

    st.caption(
        "This is an estimate learned from historical data, not a guaranteed "
        "result or a judgement of a student's ability."
    )
