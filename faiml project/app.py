# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import base64
import pyttsx3  # For AI voice narration

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Curriculum Gap Analysis", layout="wide")
st.title(" Curriculum Gap Identification Dashboard")

# =========================
# BACKGROUND IMAGE FUNCTION
# =========================
def set_bg(image_file):
    if image_file is not None:
        bg_base64 = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bg_base64}");
                background-size: cover;
                background-attachment: fixed;
            }}
            </style>
        """, unsafe_allow_html=True)

# =========================
# SIDEBAR UPLOADS
# =========================
with st.sidebar:
    st.header(" Customization")
    bg_file = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])
    video_file = st.file_uploader("Upload Demo Video", type=["mp4", "mov"])
    
    if bg_file:
        set_bg(bg_file)

# =========================
# MAIN CONTENT
# =========================
if video_file:
    st.sidebar.header(" Demo Video")
    st.sidebar.video(video_file)

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader(" Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # =========================
    # DATA VISUALIZATION SECTION
    # =========================
    st.markdown("---")
    st.header(" Data Visualizations")

    st.subheader(" Select Column for Visualization")
    selected_col = st.selectbox("Select a column to visualize", df.columns)

    if selected_col:
        col_data = df[selected_col]

        # PIE CHART
        st.subheader(" Pie Chart")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        pie_data = col_data.value_counts().head(10)
        ax_pie.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90,
                   colors=sns.color_palette("Blues"))
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

        # BAR CHART
        st.subheader(" Bar Chart")
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        bar_data = col_data.value_counts().head(10)
        sns.barplot(x=bar_data.index, y=bar_data.values, ax=ax_bar, color='#1f3b5c')
        ax_bar.set_title(f"Bar Chart of {selected_col}", fontweight='bold')
        ax_bar.set_ylabel("Count")
        ax_bar.set_xlabel(selected_col)
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)

    # =========================
    # MACHINE LEARNING ANALYSIS
    # =========================
    st.markdown("---")
    st.header(" Machine Learning Analysis")
    
    target = st.selectbox(" Select Target Column", df.columns, key="ml_target")
    features = st.multiselect(" Select Feature Columns", [col for col in df.columns if col != target])

    if target and features:
        X = df[features]
        y = df[target]

        # Encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )

            # MODEL SELECTION
            model_choice = st.selectbox(" Choose Model", ["Logistic Regression", "Random Forest"])
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=200)
            else:
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # EVALUATION
            st.subheader(" Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader(" Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            # =========================
            # PREDICTION INPUT + AI VOICE
            # =========================
            st.markdown("---")
            st.subheader(" Make a Prediction")

            input_data = {}
            cols = st.columns(2)
            for idx, col in enumerate(features):
                with cols[idx % 2]:
                    if df[col].dtype in ["int64", "float64"]:
                        val = st.number_input(f"Enter {col}", value=float(df[col].mean()))
                        input_data[col] = val
                    else:
                        options = df[col].unique().tolist()
                        val = st.selectbox(f"Select {col}", options)
                        input_data[col] = val

            if st.button("Predict", type="primary"):
                input_df = pd.DataFrame([input_data])
                input_encoded = pd.get_dummies(input_df, drop_first=True)
                input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

                prediction = model.predict(input_encoded)[0]

                # Celebration
                st.balloons()
                st.success(f" Predicted Class: {prediction}")

                # AI Voice Narration
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
                engine.setProperty('volume', 1.0)
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
                engine.say(f"The predicted class is {prediction}")
                engine.runAndWait()

                # Success Animation
                st.markdown("""
                <style>
                @keyframes celebrate {
                    0% { transform: scale(1); color: #1f77b4; }
                    50% { transform: scale(1.1); color: #ff7f0e; }
                    100% { transform: scale(1); color: #2ca02c; }
                }
                .celebrate {
                    animation: celebrate 2s ease-in-out infinite;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    padding: 20px;
                }
                </style>
                <div class="celebrate">
                     Prediction Successful!
                </div>
                """, unsafe_allow_html=True)

        except ValueError as e:
            st.error(f" Error: {e}")

# =========================
# SIDEBAR INFO
# =========================
with st.sidebar:
    st.markdown("---")
    st.markdown("###  How to Use")
    st.markdown("""
    1. **Upload** CSV data file  
    2. **View** visualizations (Line, Pie, Bar)  
    3. **Select** target & features  
    4. **Train** ML model  
    5. **Predict** outcomes  
    6. **Hear AI voice result** 🎤
    """)

    st.markdown("###  Visualization Features")
    st.markdown("""
    - Line Plot for trends  
    - Pie Chart for proportions  
    - Bar Chart for frequency  
    - Confusion Matrix for evaluation  
    """)