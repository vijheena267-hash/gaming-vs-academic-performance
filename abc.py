import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ----------------------------
# PAGE SETTINGS
# ----------------------------
st.set_page_config(
    page_title="Gaming vs Academic Performance",
    page_icon="🎮",
    layout="wide"
)

# ----------------------------
# DARK THEME
# ----------------------------
st.markdown("""
<style>
.stApp{
    background:#0A1628;
    color:white;
}

section[data-testid="stSidebar"]{
    background:#10213d;
}

div[data-testid="stMetric"]{
    background:#132746;
    padding:15px;
    border-radius:12px;
    border:1px solid #1d3d65;
}

h1,h2,h3,label,p{
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------
uploaded = st.sidebar.file_uploader(
    "Upload CSV",
    type="csv"
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("Gaming_Academic_Performance Dataset.csv")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("🎮 GameXAcademics")

menu = st.sidebar.radio(
    "Menu",
    [
        "Overview",
        "Gaming Behavior",
        "Academic Performance",
        "Health & Wellbeing",
        "Correlations",
        "Data Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

gender = st.sidebar.selectbox(
    "Gender",
    ["All"] + sorted(df["gender"].unique())
)

genre = st.sidebar.selectbox(
    "Gaming Genre",
    ["All"] + sorted(df["gaming_genre"].unique())
)

stress = st.sidebar.selectbox(
    "Stress Level",
    ["All"] + sorted(df["stress_level"].unique())
)

# ----------------------------
# APPLY FILTERS
# ----------------------------
filtered = df.copy()

if gender != "All":
    filtered = filtered[
        filtered["gender"] == gender
    ]

if genre != "All":
    filtered = filtered[
        filtered["gaming_genre"] == genre
    ]

if stress != "All":
    filtered = filtered[
        filtered["stress_level"] == stress
    ]

# ----------------------------
# HEADER
# ----------------------------
st.title("🎮 Gaming vs Academic Performance")

st.write(
    "Explore how gaming habits affect grades, study hours, stress, sleep and overall academic performance."
)

# ----------------------------
# KPI CARDS
# ----------------------------
c1,c2,c3,c4,c5,c6 = st.columns(6)

c1.metric(
    "Students",
    len(filtered)
)

c2.metric(
    "Average Grade",
    round(filtered["grades"].mean(),1)
)

c3.metric(
    "Gaming Hours",
    round(filtered["gaming_hours"].mean(),1)
)

c4.metric(
    "Study Hours",
    round(filtered["study_hours"].mean(),1)
)

c5.metric(
    "Addiction Score",
    round(filtered["addiction_score"].mean(),1)
)

stress_map={
    "Low":1,
    "Medium":2,
    "High":3
}

filtered["Stress Score"]=filtered["stress_level"].map(stress_map)

c6.metric(
    "Average Stress",
    round(filtered["Stress Score"].mean(),1)
)
# ======================================================
# OVERVIEW
# ======================================================

if menu == "Overview":

    st.subheader("Overview")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered,
            x="grades",
            nbins=10,
            color_discrete_sequence=["deepskyblue"],
            title="Grade Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            filtered,
            names="gender",
            hole=0.6,
            title="Gender Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    genre = filtered.groupby("gaming_genre").size().reset_index(name="Students")

    fig = px.bar(
        genre,
        x="gaming_genre",
        y="Students",
        color="gaming_genre",
        title="Gaming Genre Popularity"
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# GAMING BEHAVIOR
# ======================================================

elif menu == "Gaming Behavior":

    st.subheader("Gaming Behavior")

    col1, col2 = st.columns(2)

    # Histogram
    with col1:

        fig = px.histogram(
            filtered,
            x="gaming_hours",
            nbins=10,
            color_discrete_sequence=["orange"],
            title="Gaming Hours Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Device Usage
    with col2:

        fig = px.histogram(
            filtered,
            x="device_usage",
            nbins=10,
            color_discrete_sequence=["violet"],
            title="Device Usage Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    # Scatter Plot
    with col3:

        fig = px.scatter(
            filtered,
            x="gaming_hours",
            y="grades",
            color="gender",
            title="Gaming Hours vs Grades"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Bubble Chart
    with col4:

        fig = px.scatter(
            filtered,
            x="gaming_hours",
            y="addiction_score",
            size="Stress Score",
            color="stress_level",
            title="Gaming Hours vs Addiction Score"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col5, col6 = st.columns(2)

    # Violin Plot
    with col5:

        fig = px.violin(
            filtered,
            x="gaming_genre",
            y="addiction_score",
            color="gaming_genre",
            box=True,
            title="Addiction Score by Genre"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Area Chart
    with col6:

        gaming = filtered.sort_values("gaming_hours")

        fig = px.area(
            gaming,
            x="gaming_hours",
            y="grades",
            title="Gaming Hours Trend"
        )

        st.plotly_chart(fig, use_container_width=True)
        # ======================================================
# ACADEMIC PERFORMANCE
# ======================================================

elif menu == "Academic Performance":

    st.subheader("Academic Performance")

    col1, col2 = st.columns(2)

    # Study Hours vs Grades
    with col1:
        fig = px.scatter(
            filtered,
            x="study_hours",
            y="grades",
            color="gender",
            title="Study Hours vs Grades"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Attendance vs Grades
    with col2:
        fig = px.scatter(
            filtered,
            x="attendance",
            y="grades",
            color="gaming_genre",
            title="Attendance vs Grades"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    # Study Hours Histogram
    with col3:
        fig = px.histogram(
            filtered,
            x="study_hours",
            nbins=10,
            color_discrete_sequence=["deepskyblue"],
            title="Study Hours Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Average Grade by Genre
    with col4:
        avg = filtered.groupby("gaming_genre")["grades"].mean().reset_index()

        fig = px.bar(
            avg,
            x="gaming_genre",
            y="grades",
            color="gaming_genre",
            title="Average Grade by Genre"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Average Grade by Age
    age = filtered.groupby("age")["grades"].mean().reset_index()

    fig = px.line(
        age,
        x="age",
        y="grades",
        markers=True,
        title="Average Grade by Age"
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# HEALTH & WELLBEING
# ======================================================

elif menu == "Health & Wellbeing":

    st.subheader("Health & Wellbeing")

    col1, col2 = st.columns(2)

    # Sleep vs Stress
    with col1:
        fig = px.box(
            filtered,
            x="stress_level",
            y="sleep_hours",
            color="stress_level",
            title="Sleep Hours by Stress Level"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Reaction Time
    with col2:
        fig = px.scatter(
            filtered,
            x="gaming_hours",
            y="reaction_time_ms",
            color="stress_level",
            title="Reaction Time vs Gaming Hours"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    # Stress Distribution
    with col3:
        fig = px.histogram(
            filtered,
            x="stress_level",
            color="stress_level",
            title="Stress Level Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Grades by Stress
    with col4:
        fig = px.box(
            filtered,
            x="stress_level",
            y="grades",
            color="stress_level",
            title="Grades by Stress Level"
        )

        st.plotly_chart(fig, use_container_width=True)


# ======================================================
# CORRELATIONS
# ======================================================

elif menu == "Correlations":

    st.subheader("Correlation Matrix")

    numeric = filtered.select_dtypes(include="number")

    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# DATA EXPLORER
# ======================================================

elif menu == "Data Explorer":

    st.subheader("Dataset")

    st.dataframe(filtered, use_container_width=True)

    st.markdown("### Summary Statistics")

    st.dataframe(
        filtered.describe(),
        use_container_width=True
    )

    st.download_button(
        "Download Filtered Dataset",
        filtered.to_csv(index=False),
        file_name="Gaming_Data.csv",
        mime="text/csv"
    )
    # ==========================================
# DASHBOARD HEADER
# ==========================================

st.markdown("""
<div style='background:#10213d;
padding:20px;
border-radius:12px;
border-left:5px solid cyan;
margin-bottom:20px;'>

<h1 style='color:white;margin:0;'>
🎮 Gaming vs Academic Performance
</h1>

<p style='color:lightgray;'>
Explore how gaming habits influence study hours,
grades, stress level and student wellbeing.
</p>

</div>
""", unsafe_allow_html=True)

col1,col2,col3,col4=st.columns(4)

col1.info(f"👨‍🎓 Students : {len(filtered)}")
col2.info(f"📊 Columns : {len(filtered.columns)}")
col3.info(f"🎮 Genres : {filtered['gaming_genre'].nunique()}")
col4.info("📅 Live Dashboard")