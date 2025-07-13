import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# Load & Preprocess Dataset
# ------------------------------
@st.cache_data
def load_data():
    df_raw = pd.read_csv("insurance.csv")  # data asli
    df_pre = pd.read_csv("insurance_preprocessed.csv")  # data preprocessing

    # Gabungkan dua dataset berdasarkan indeks
    df_combined = df_pre.copy()
    df_combined['sex_original'] = df_raw['sex']
    df_combined['smoker_original'] = df_raw['smoker']

    # Encode hanya kolom smoker untuk model
    le_smoker = LabelEncoder()
    df_combined['smoker_encoded'] = le_smoker.fit_transform(df_combined['smoker_original'])

    label_encoders = {
        'smoker': le_smoker
    }

    return df_combined, label_encoders

df, label_encoders = load_data()

# ------------------------------
# Latih Model
# ------------------------------
@st.cache_resource
def train_model(data):
    X = data[['age', 'bmi', 'children', 'smoker_encoded']]
    y = data['charges']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Prediksi Biaya Asuransi", layout="centered")
st.title("💰 Prediksi Biaya Asuransi")
st.markdown("Masukkan informasi berikut untuk memprediksi biaya asuransi kesehatan:")

# Form Input
age = st.slider("Usia", 18, 100, 30)
sex = st.selectbox("Jenis Kelamin", df['sex_original'].unique())  # untuk tampilan saja
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Jumlah Anak", 0, 5, 0)
smoker = st.selectbox("Perokok", df['smoker_original'].unique())  # untuk tampilan saja

# Encode input
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'smoker_encoded': [label_encoders['smoker'].transform([smoker])[0]]
})

# Prediksi
if st.button("🔍 Prediksi Charges"):
    prediction = model.predict(input_data)[0]
    st.success(f"💸 Prediksi Biaya Asuransi: **${prediction:,.2f}**")

# Optional: tampilkan data sample
with st.expander("📊 Lihat Sample Dataset"):
    st.dataframe(df[['age', 'sex_original', 'bmi', 'children', 'smoker_original', 'charges']].head())

st.markdown("---")
st.caption("Aplikasi prediksi dengan Random Forest – Menampilkan data asli (tanpa encoding di UI)")
