# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Load data ---
data_path = "../data/processed/cleaned_citizen_feedback.csv"
df = pd.read_csv(data_path)

# --- Gemini Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # make sure to set this in your env
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
else:
    st.warning("Gemini API key not found. Please set GEMINI_API_KEY in your environment.")

# --- App Layout ---
st.set_page_config(page_title="InsightNation Dashboard", layout="wide")
st.title("\ud83c\udf0d InsightNation: Citizen Feedback Dashboard")

st.sidebar.header("Filters")
city_filter = st.sidebar.multiselect("Select Cities", options=df["city"].unique(), default=df["city"].unique())
age_filter = st.sidebar.multiselect("Select Age Groups", options=df["age_group"].unique(), default=df["age_group"].unique())

filtered_df = df[(df["city"].isin(city_filter)) & (df["age_group"].isin(age_filter))]

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Responses", len(filtered_df))
col2.metric("Unique Cities", filtered_df["city"].nunique())
col3.metric("Avg Park Visit Frequency", filtered_df["park_visit_freq"].mode()[0])

# --- Visualizations ---
st.subheader("\ud83c\udf0e City-wise Service Satisfaction")
fig1 = px.histogram(filtered_df, x="city", color="local_service_satisfaction",
                    barmode="group", title="Service Satisfaction by City")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("\ud83c\udfdb\ufe0f Library Satisfaction")
fig2 = px.pie(filtered_df, names="library_satisfaction", title="Library Satisfaction Distribution")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("\ud83c\udf3f Transport Safety by Gender")
fig3 = px.histogram(filtered_df, x="gender", color="transport_safety", barmode="group")
st.plotly_chart(fig3, use_container_width=True)

# --- WordCloud ---
st.subheader("\u2728 WordCloud of Suggestions")
text_cols = ["transport_suggestions", "park_suggestions", "library_suggestions", "local_service_suggestions"]
all_text = " ".join(filtered_df[col].fillna("") for col in text_cols)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# --- Gemini AI Insights ---
st.subheader("\ud83e\udd16 Gemini AI Insights")
if GEMINI_API_KEY:
    sample_data = filtered_df[text_cols].dropna().sample(5).astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
    prompt = "Analyze the following citizen feedback and give strategic insights for city service improvements:\n" + "\n\n".join(sample_data)

    if st.button("Generate AI Insights"):
        with st.spinner("Thinking with Gemini..."):
            try:
                response = model.generate_content(prompt)
                st.success("Here's what Gemini suggests:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error generating response: {e}")
else:
    st.info("Enable Gemini API to access AI-generated insights.")

# --- Footer ---
st.markdown("---")
st.markdown("\u00a9 2025 InsightNation | Built with Streamlit & Gemini AI")
