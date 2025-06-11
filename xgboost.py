import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from xgboost import XGBClassifier  # ✅ XGBoost

st.set_page_config(page_title="Mental Health Treatment Prediction", layout="centered")
st.title("🧠 Mental Health Treatment Prediction App")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("mental_health_dataset.csv")
    df.drop(columns=["Timestamp", "Occupation"], errors='ignore', inplace=True)

    df['self_employed'] = df['self_employed'].fillna('Unknown')
    df['family_history'] = df['family_history'].fillna('No')
    df['mental_health_interview'] = df['mental_health_interview'].fillna('Maybe')
    df['care_options'] = df['care_options'].fillna('Maybe')
    df['Gender'] = df['Gender'].fillna('Other')
    df['Country'] = df['Country'].fillna('United States')

    binary_cols = [
        'self_employed', 'family_history', 'treatment', 'Growing_Stress',
        'Changes_Habits', 'Mental_Health_History', 'Coping_Struggles',
        'Work_Interest', 'Social_Weakness'
    ]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Unknown': -1})

    df['mental_health_interview'] = df['mental_health_interview'].map({'Yes': 1, 'Maybe': 0.5, 'No': 0, 'Not sure': 0.5})
    df['care_options'] = df['care_options'].map({'Yes': 1, 'Maybe': 0.5, 'No': 0, 'Not sure': 0.5})
    df['Mood_Swings'] = df['Mood_Swings'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['Days_Indoors'] = df['Days_Indoors'].map({'1-14 days': 1, '15-30 days': 2, '30+ days': 3})

    df = pd.get_dummies(df, columns=['Gender', 'Country'])

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)

    y = df['treatment']
    X = df.drop(columns=['treatment'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=20)

    # ✅ Use XGBoost instead of Logistic Regression
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("📊 Model Evaluation Metrics")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"]
    })
    st.table(metrics_df.style.set_properties(**{
        'background-color': '#f0f8ff',
        'color': 'black',
        'border-color': 'lightgrey',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4682b4'), ('color', 'white'), ('text-align', 'center')]}
    ]))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=["Actual: No Treatment", "Actual: Treatment"],
                         columns=["Predicted: No Treatment", "Predicted: Treatment"])

    st.markdown("### 📉 Confusion Matrix")
    st.dataframe(cm_df.style.set_properties(**{
        'background-color': '#fffaf0',
        'color': 'black',
        'border-color': 'grey',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#cd5c5c'), ('color', 'white'), ('text-align', 'center')]}
    ]))

    return model, scaler, X.columns

model, scaler, feature_columns = load_and_train()