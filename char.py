import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import xgboost as xg

# ১. পেজ কনফিগারেশন ও থিম (সবচেয়ে ওপরে থাকতে হবে)
st.set_page_config(
    page_title="Workforce Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

# সিএসএস (CSS) দিয়ে অ্যাপের ডিজাইন সুন্দর করা
st.markdown(
    """
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .kpi-box { padding: 20px; background-color: #F3F4F6; border-radius: 10px; text-align: center; border-left: 5px solid #1E3A8A; }
    </style>
""",
    unsafe_allow_html=True,
)


# ২. ডাটা লোড ও ক্যাশ ফাংশন
@st.cache_data
def load_data():
  # আপনার লোকাল ফাইল পাথ
  data = pd.read_csv("client_churn_messy.csv")
  df = pd.DataFrame(data)

  # ডাটা ক্লিনিং পাইপলাইন
  df.loc[0, ["Support_Calls", "Total_Spent", "Region"]] = df.loc[
      0, ["Region", "Support_Calls", "Total_Spent"]
  ].values
  df["Support_Calls"] = pd.to_numeric(
      df["Support_Calls"], errors="coerce"
  )
  df["Support_Calls"] = df["Support_Calls"].fillna(
      df["Support_Calls"].median()
  )
  df.loc[7, "Total_Spent"] = "130"
  df["Churned"] = df["Churned"].fillna("Yes")
  df["Churned"] = df["Churned"].map({"N": "No"})
  df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
  df["Age"] = df["Age"].fillna(df["Age"].median())
  df["Churned"] = df["Churned"].fillna("Yes")

  df["Total_Spent"] = pd.to_numeric(
      df["Total_Spent"].astype(str).str.replace("$", "", regex=False),
      errors="coerce",
  ).replace(-50, np.nan)
  df["Total_Spent"] = df["Total_Spent"].fillna(
      df["Total_Spent"].median()
  )
  df["Region"] = df["Region"].fillna("UK-London")
  df["Churned"] = df["Churned"].map({"Yes": 1, "No": 0})
  return df


df = load_data()

# ৩. সাইডবার নেভিগেশন এবং ফিল্টার
st.sidebar.image(
    "https://flaticon.com", width=80
)
st.sidebar.title("Navigation & Filters")
page = st.sidebar.radio(
    "Select Screen:", ["📊 Analytics Overview", "🤖 ML Churn Predictions"]
)

st.sidebar.markdown("---")
selected_region = st.sidebar.multiselect(
    "Filter by Region:",
    options=df["Region"].unique(),
    default=df["Region"].unique(),
)

# ফিল্টার অনুযায়ী ডাটা আপডেট
filtered_df = df[df["Region"].isin(selected_region)]

# ৪. স্ক্রিন ১: অ্যানালিটিক্স ওভারভিউ
if page == "📊 Analytics Overview":
  st.markdown(
      "<div class='main-title'>💼 Enterprise Workforce Analytics</div>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<div class='sub-title'>Monitor employee metrics, spending behaviors, and data insights live.</div>",
      unsafe_allow_html=True,
  )

  # KPI কার্ড ডিজাইন
  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label="Total Employees Monitored", value=len(filtered_df))
  with col2:
    st.metric(
        label="Average Employee Age", value=f"{filtered_df['Age'].mean():.1f}"
    )
  with col3:
    st.metric(
        label="Total Budget Spent ($)",
        value=f"{filtered_df['Total_Spent'].sum():,.2f}",
    )

  st.markdown("### 📈 Visual Data Insights")
  c1, c2 = st.columns(2)

  with c1:
    st.write("**Age vs Total Spent (Interactive Scatter Chart)**")
    fig = px.scatter(
        filtered_df,
        x="Age",
        y="Total_Spent",
        color="Subscription_Type",
        size="Support_Calls",
        hover_data=["Region"],
    )
    st.plotly_chart(fig, use_container_width=True)

  with c2:
    st.write("**Total Spent Distribution by Region (Bar Chart)**")
    fig2 = px.bar(
        filtered_df,
        x="Region",
        y="Total_Spent",
        color="Subscription_Type",
        barmode="group",
    )
    st.plotly_chart(fig2, use_container_width=True)

  # ডাটা টেবিল ভিউ
  with st.expander("🔍 Click to Inspect Cleaned Dataset Rows"):
    st.dataframe(filtered_df, use_container_width=True)

# ৫. স্ক্রিন ২: মেশিন লার্নিং প্রেডিকশন
else:
  st.markdown(
      "<div class='main-title'>🤖 Predictive Risk Intelligence AI</div>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<div class='sub-title'>XGBoost Supervised Learning classification pipeline detecting worker turnover.</div>",
      unsafe_allow_html=True,
  )

  # মডেল প্রিপারেশন এবং ট্রেইনিং
  X = df.drop(
      columns=["Client_ID", "Churned", "Join_Date", "Last_Active"],
      errors="ignore",
  )
  y = df["Churned"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=55
  )

  numerical_features = ["Age", "Support_Calls", "Total_Spent"]
  numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

  categorical_features = ["Subscription_Type", "Region"]
  categorical_transformer = Pipeline(
      steps=[
          (
              "encoder",
              OneHotEncoder(handle_unknown="ignore", sparse_output=False),
          )
      ]
  )

  transformer = ColumnTransformer(
      transformers=[
          ("num", numerical_transformer, numerical_features),
          ("cate", categorical_transformer, categorical_features),
      ]
  )

  full_pipeline = Pipeline(
      steps=[
          ("transformer", transformer),
          (
              "xg",
              xg.XGBClassifier(
                  n_estimators=100,
                  learning_rate=0.1,
                  max_depth=3,
                  random_state=42,
              ),
          ),
      ]
  )

  full_pipeline.fit(X_train, y_train)
  prediction = full_pipeline.predict(X_test)

  # প্রেডিকশন এলার্ট ইউজার ইন্টারফেস
  st.markdown("### 🔔 Automated Attrition Risk Assessment")
  st.write(
      "The AI model evaluated the test subset workers. Here are the immediate"
      " risk alerts:"
  )

  for idx, pred in enumerate(prediction):
    if pred == 1:
      st.error(
          f"⚠️ **Worker Profile #{idx + 1}**: High Risk Detection! The model"
          " predicts this worker will leave permanently."
      )
    else:
      st.success(
          f"✅ **Worker Profile #{idx + 1}**: Safe. This worker is predicted to"
          " remain with the company."
      )