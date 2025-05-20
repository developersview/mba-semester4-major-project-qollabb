# 📂 app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
# --- Load data ---
data_path = os.getenv("cleaned_csv_path")
df = pd.read_csv(data_path)

# ----------------------------
# Setup Google Gemini API
# ----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-001")

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="InsightNation Dashboard", layout="wide")
st.title("InsightNation - Government Data Analytics Platform")




# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# ----------------------------
# File Upload Section
# ----------------------------
def upload_dataset():
    uploaded_file = st.file_uploader("Upload your cleaned citizen feedback CSV:", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
        st.dataframe(df.head(20))
        st.session_state["df"] = df

# ----------------------------
# EDA and Feedback Insights
# ----------------------------
def citizen_feedback_insights():
    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    if st.button("Generate Feedback Insights"):
        st.subheader("Key Feedback Statistics")
        
        # avg_scores = {
        #     "Toilet Cleanliness": df["toilet_cleanliness"].mean(),
        #     "Transport Satisfaction": df["transport_satisfaction"].mean(),
        #     "Library Satisfaction": df["library_satisfaction"].mean(),
        #     "Local Service Satisfaction": df["local_service_satisfaction"].mean()
        # }
        # st.write(avg_scores)

        st.subheader("AI-Generated Insight Summary")
        sample_text = " ".join(df["local_service_suggestions"].dropna().astype(str).sample(10))
        if sample_text:
            prompt = f"Summarize key citizen feedback themes from the following text: {sample_text}. Also, provide a sentiment score (positive, negative, neutral) for each theme."
            response = model.generate_content(prompt)
            st.success(response.text)

# # ----------------------------
# # Citizen Interaction Chatbot
# # ----------------------------
# def citizen_chatbot():
#     st.header("BizMate - Your Business Chatbot")
#     st.markdown("Ask questions about business strategy, analytics, or case study scenarios")

#     # if "chat_historty" not in st.session_state:
#     #     st.session_state.chat_history = []

#     user_input = st.chat_input("Ask your virtual business consultant...")


#     if user_input:
#         chat_prompt = """ You are a business strategy consultant. Use previous conversation context to give informed answers"""
#         history = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.chat_history])
        
#         full_prompt = f""" 
#             {chat_prompt} 
#             {history}
#             User: {user_input}
#             AI:
#         """

#         response = model.generate_content(full_prompt)
#         reply = response.text

#         st.session_state.chat_history.append({
#             "user": user_input,
#             "ai": reply
#         })

#     for idx, msg in enumerate(st.session_state.chat_history):
#         with st.chat_message("user"):
#             st.markdown(msg["user"])
#         with st.chat_message("ai"):
#             st.markdown(msg["ai"])

# ----------------------------
# AI Policy Advisor
# ----------------------------
def ai_policy_advisor():
    st.subheader("AI Policy Advisor for Public Services")
    scenario = st.text_area("Describe a public service scenario:")

    if st.button("Generate Strategies"):
        if scenario:
            prompt = f"Suggest 3 detailed strategies to improve public service based on this scenario: {scenario}"
            response = model.generate_content(prompt)
            st.success(response.text)

# ----------------------------
# Visual Analytics Dashboard
# ----------------------------
def visual_dashboard():
    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    st.subheader("Visual Analytics")

    st.sidebar.header("Filters")
    city_filter = st.sidebar.multiselect("Select Cities", options=df["city"].unique(), default=df["city"].unique())
    age_filter = st.sidebar.multiselect("Select Age Groups", options=df["age_group"].unique(), default=df["age_group"].unique())

    filtered_df = df[(df["city"].isin(city_filter)) & (df["age_group"].isin(age_filter))]

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responses", len(filtered_df))
    col2.metric("Unique Cities", filtered_df["city"].nunique())
    col3.metric("Age Group", filtered_df["age_group"].nunique())
    col4.metric("Avg Serive Use Frequency", filtered_df["service_use_freq"].mode()[0])


    # --- Visualizations ---
    st.subheader("City-wise Service Satisfaction")
    fig1 = px.histogram(filtered_df, x="city", color="local_service_satisfaction",
                        barmode="group", title="Service Satisfaction by City")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Library Satisfaction")
    fig2 = px.pie(filtered_df, names="library_satisfaction", title="Library Satisfaction Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Transport Safety by Gender")
    fig3 = px.histogram(filtered_df, x="gender", color="transport_safety", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📊 Age Group Distribution")
    fig_age = px.histogram(
        filtered_df,
        x='age_group',
        category_orders={"age_group": df['age_group'].value_counts().index.tolist()},
        title="Age Group Distribution",
        color='age_group'
    )
    fig_age.update_layout(xaxis_title="Age Group", yaxis_title="Count", bargap=0.2)
    st.plotly_chart(fig_age, use_container_width=True)

    # 2. Gender distribution
    st.subheader("📊 Gender Distribution")
    fig_gender = px.histogram(
        filtered_df,
        x='gender',
        title="Gender Distribution",
        color= 'gender'
    )
    fig_gender.update_layout(xaxis_title="Gender", yaxis_title="Count", bargap=0.2)
    st.plotly_chart(fig_gender, use_container_width=True)

    # 3. City distribution
    st.subheader("📊 City-wise Feedback Count")
    fig_city = px.histogram(
        filtered_df,
        x='city',
        category_orders={"city": df['city'].value_counts().index.tolist()},
        title="City-wise Feedback Count",
        color_discrete_sequence=["#00CC96"]
    )
    fig_city.update_layout(xaxis_title="City", yaxis_title="Count", bargap=0.2)
    fig_city.update_xaxes(tickangle=45)
    st.plotly_chart(fig_city, use_container_width=True)

    # 4. Service Satisfaction Distribution
    st.subheader("📊 Toilet Service Satisfaction Levels")
    fig_service = px.histogram(
        filtered_df,
        x='toilet_cleanliness',
        title="Local Service Satisfaction Levels",
        color='toilet_cleanliness'
    )
    fig_service.update_layout(xaxis_title="Satisfaction Level", yaxis_title="Count", bargap=0.2)
    st.plotly_chart(fig_service, use_container_width=True)


    # --- WordCloud ---
    st.subheader("WordCloud of Suggestions")
    #text_cols = ["transport_suggestions", "park_suggestions", "library_suggestions", "local_service_suggestions"]
    #all_text = " ".join(filtered_df[col].fillna("") for col in text_cols)
    text = ' '.join(df['local_service_suggestions'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    
# ----------------------------
# Sentiment SWOT Analysis
# ----------------------------
def sentiment_swot():
    df = st.session_state.get("df")
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    st.subheader("SWOT Analysis from Feedback")
    if st.button("Generate SWOT Analysis"):
        sample_feedback = " ".join(df["local_service_suggestions"].dropna().astype(str).sample(10))

        prompt = f"Based on the following citizen feedback, generate a SWOT analysis and Also Gerate a 2x2 table of SWOT:\n\n{sample_feedback}"
        response = model.generate_content(prompt)

        st.success(response.text)

# ----------------------------
# Sidebar Navigation
# ----------------------------

# # Sidebar Navigation
# page = st.sidebar.radio("Go to", [
#     "📂 Upload New Dataset",
#     "📈 Citizen Feedback Insights", 
#     "📊 Visual Analytics Dashboard",  
#     "🧩 Sentiment SWOT Analysis",
#     "⚙️ AI Policy Advisor"
    
# ])

# if page == "📂 Upload New Dataset":
#     upload_dataset()
# elif page == "📈 Citizen Feedback Insights":
#     citizen_feedback_insights()
# elif page == "⚙️ AI Policy Advisor":
#     ai_policy_advisor()
# elif page == "📊 Visual Analytics Dashboard":
#     visual_dashboard()
# elif page == "🧩 Sentiment SWOT Analysis":
#     sentiment_swot()


# ----------------------------
# Sidebar Navigation
# ----------------------------
menu = st.sidebar.selectbox(
    "Navigate",
    [
        "📂 Upload New Dataset",
        "📈 Citizen Feedback Insights",
        "📊 Visual Analytics Dashboard",
        "🧩 Sentiment SWOT Analysis",
        "🤖 AI Policy Advisor",
    ]
)

if menu == "📂 Upload New Dataset":
    upload_dataset()
elif menu == "📈 Citizen Feedback Insights":
    citizen_feedback_insights()
elif menu == "🤖 AI Policy Advisor":
    ai_policy_advisor()
elif menu == "📊 Visual Analytics Dashboard":
    visual_dashboard()
elif menu == "🧩 Sentiment SWOT Analysis":
    sentiment_swot()

st.sidebar.info("©️ 2025 InsightNation. All rights reserved.")