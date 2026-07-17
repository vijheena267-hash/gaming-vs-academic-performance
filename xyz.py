import streamlit as st
import pandas as pd
import plotly.express as px

#
st.set_page_config(
    page_title="Gaming vs Academic Performance",
    page_icon="🎮",
    layout="wide"
)


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


uploaded = st.sidebar.file_uploader(
    "Upload CSV File",
    type="csv"
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("Gaming_Academic_Performance Dataset.csv")



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



stress_map = {
    "Low":1,
    "Medium":2,
    "High":3
}

filtered["Stress Score"] = filtered["stress_level"].map(stress_map)



st.markdown("""
<div style='background:#10213d;
padding:20px;
border-radius:12px;
border-left:6px solid cyan;
margin-bottom:20px;'>

<h1 style='color:white;margin:0;'>
🎮 Gaming vs Academic Performance Dashboard
</h1>

<p style='color:lightgray;font-size:17px;'>
Explore how gaming habits influence study hours,
grades, stress level and student wellbeing.
</p>

</div>
""", unsafe_allow_html=True)


info1,info2,info3,info4 = st.columns(4)

info1.info(f"👨‍🎓 Students : {len(filtered)}")
info2.info(f"📊 Columns : {len(filtered.columns)}")
info3.info(f"🎮 Genres : {filtered['gaming_genre'].nunique()}")
info4.info("📅 Live Dashboard")


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

c6.metric(
    "Average Stress",
    round(filtered["Stress Score"].mean(),1)
)


if menu == "Overview":

    st.subheader("📊 Dashboard Overview")

    

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered,
            x="grades",
            nbins=10,
            color_discrete_sequence=["deepskyblue"],
            title="Grade Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            filtered,
            names="gender",
            hole=0.5,
            title="Gender Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

   

    genre_data = filtered.groupby(
        "gaming_genre"
    ).size().reset_index(name="Students")

    fig = px.bar(
        genre_data,
        x="gaming_genre",
        y="Students",
        color="gaming_genre",
        title="Gaming Genre Popularity"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A1628",
        plot_bgcolor="#0A1628"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
   

elif menu == "Gaming Behavior":

    st.subheader("🎮 Gaming Behavior")

   

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered,
            x="gaming_hours",
            nbins=10,
            color_discrete_sequence=["orange"],
            title="Gaming Hours Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(
            filtered,
            x="device_usage",
            nbins=10,
            color_discrete_sequence=["violet"],
            title="Device Usage Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    

    col3, col4 = st.columns(2)

    with col3:

        fig = px.scatter(
            filtered,
            x="gaming_hours",
            y="grades",
            color="gender",
            title="Gaming Hours vs Grades"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        fig = px.scatter(
            filtered,
            x="gaming_hours",
            y="addiction_score",
            color="stress_level",
            size="Stress Score",
            title="Gaming Hours vs Addiction Score"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    

    col5, col6 = st.columns(2)

    with col5:

        fig = px.violin(
            filtered,
            x="gaming_genre",
            y="addiction_score",
            color="gaming_genre",
            box=True,
            title="Addiction Score by Genre"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col6:

        gaming = filtered.sort_values("gaming_hours")

        fig = px.area(
            gaming,
            x="gaming_hours",
            y="grades",
            title="Gaming Hours Trend"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)
        
elif menu == "Academic Performance":

    st.subheader("📚 Academic Performance")

    

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            filtered,
            x="study_hours",
            y="grades",
            color="gender",
            title="Study Hours vs Grades"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.scatter(
            filtered,
            x="attendance",
            y="grades",
            color="gaming_genre",
            title="Attendance vs Grades"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    
    col3, col4 = st.columns(2)

    with col3:

        fig = px.histogram(
            filtered,
            x="study_hours",
            nbins=10,
            color_discrete_sequence=["deepskyblue"],
            title="Study Hours Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        avg_grade = filtered.groupby(
            "gaming_genre"
        )["grades"].mean().reset_index()

        fig = px.bar(
            avg_grade,
            x="gaming_genre",
            y="grades",
            color="gaming_genre",
            title="Average Grade by Gaming Genre"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

   
    age_grade = filtered.groupby(
        "age"
    )["grades"].mean().reset_index()

    fig = px.line(
        age_grade,
        x="age",
        y="grades",
        markers=True,
        title="Average Grade by Age"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A1628",
        plot_bgcolor="#0A1628"
    )

    st.plotly_chart(fig, use_container_width=True)
    

elif menu == "Health & Wellbeing":

    st.subheader("❤️ Health & Wellbeing")

    
    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            filtered,
            x="stress_level",
            y="sleep_hours",
            color="stress_level",
            title="Sleep Hours by Stress Level"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.scatter(
            filtered,
            x="gaming_hours",
            y="reaction_time_ms",
            color="stress_level",
            title="Reaction Time vs Gaming Hours"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --------------------------------
    # Histogram & Box Plot
    # --------------------------------

    col3, col4 = st.columns(2)

    with col3:

        fig = px.histogram(
            filtered,
            x="stress_level",
            color="stress_level",
            title="Stress Level Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        fig = px.box(
            filtered,
            x="stress_level",
            y="grades",
            color="stress_level",
            title="Grades by Stress Level"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628"
        )

        st.plotly_chart(fig, use_container_width=True)
        # ===================================
# CORRELATIONS
# ===================================

elif menu == "Correlations":

    st.subheader("🔥 Correlation Matrix")

    # Select only numeric columns
    numeric = filtered.select_dtypes(include="number")

    # Correlation
    corr = numeric.corr()

    # Heatmap
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A1628",
        plot_bgcolor="#0A1628"
    )

    st.plotly_chart(fig, use_container_width=True)


# ===================================
# DATA EXPLORER
# ===================================

elif menu == "Data Explorer":

    st.subheader("📂 Dataset")

    st.dataframe(
        filtered,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📊 Summary Statistics")

    st.dataframe(
        filtered.describe(),
        use_container_width=True
    )

    st.markdown("---")

    st.download_button(
        label="⬇ Download Filtered Dataset",
        data=filtered.to_csv(index=False),
        file_name="Gaming_Data.csv",
        mime="text/csv"
    )