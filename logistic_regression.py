import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

st.set_page_config(page_title="Mental Health Treatment Prediction", layout="centered")

st.title("🧠 Mental Health Treatment Prediction App")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("mental_health_dataset.csv")

    df.drop(columns=["Timestamp", "Occupation"], errors='ignore', inplace=True)

    df = df.copy()
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

    model = LogisticRegression()
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

    # Hardcoded Confusion Matrix for 83% accuracy with 292365 records
    import numpy as np
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

st.subheader("📾 Predict Mental Health Treatment Need")

user_input = {}

user_input["self_employed"] = st.selectbox("Are you self-employed?", ["Yes", "No", "Unknown"])
user_input["family_history"] = st.selectbox("Family history of mental illness?", ["Yes", "No"])
user_input["Growing_Stress"] = st.selectbox("Are you experiencing growing stress?", ["Yes", "No"])
user_input["Changes_Habits"] = st.selectbox("Changed habits recently?", ["Yes", "No"])
user_input["Mental_Health_History"] = st.selectbox("Do you have a mental health history?", ["Yes", "No"])
user_input["Mood_Swings"] = st.selectbox("Mood swings level?", ["Low", "Medium", "High"])
user_input["Coping_Struggles"] = st.selectbox("Struggling to cope?", ["Yes", "No"])
user_input["Work_Interest"] = st.selectbox("Lost interest in work?", ["Yes", "No"])
user_input["Social_Weakness"] = st.selectbox("Facing social weakness?", ["Yes", "No"])
user_input["mental_health_interview"] = st.selectbox("Would you discuss mental health at interview?", ["Yes", "Maybe", "No", "Not sure"])
user_input["care_options"] = st.selectbox("Are care options available at your workplace?", ["Yes", "Maybe", "No", "Not sure"])
user_input["Days_Indoors"] = st.selectbox("How many days indoors recently?", ["1-14 days", "15-30 days", "30+ days"])
user_input["Gender"] = st.selectbox("Gender", ["Female", "Male", "Other"])
user_input["Country"] = st.selectbox("Country", ["United States", "Canada", "United Kingdom", "Poland", "Australia", "South Africa"])

mapped = {
    "Yes": 1, "No": 0, "Maybe": 0.5, "Not sure": 0.5, "Unknown": -1,
    "Low": 0, "Medium": 1, "High": 2,
    "1-14 days": 1, "15-30 days": 2, "30+ days": 3
}
for key in user_input:
    if key in ["Mood_Swings", "Days_Indoors", "mental_health_interview", "care_options"]:
        user_input[key] = mapped[user_input[key]]
    elif user_input[key] in mapped:
        user_input[key] = mapped[user_input[key]]

input_df = pd.DataFrame([user_input])

for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

input_scaled = scaler.transform(input_df)

if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    result = "✅ The person is likely to seek **mental health treatment**." if prediction == 1 else "❌ The person is **not likely** to seek mental health treatment."
    st.success(result)
