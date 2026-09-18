import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Mental Health in Tech Survey", layout="wide")

# ---------------------------
# Load and clean data (cached so it only runs once)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("survey.csv")

    # Clean Age
    df = df[(df['Age'] >= 18) & (df['Age'] <= 75)]

    # Clean Gender
    df['Gender'] = df['Gender'].str.strip().str.lower()
    male_terms = ['male', 'm', 'man', 'cis male', 'male (cis)', 'cis man',
                  'mail', 'malr', 'make', 'msle', 'guy (-ish) ^_^', 'male-ish']
    female_terms = ['female', 'f', 'woman', 'cis female', 'cis-female/femme',
                     'female (cis)', 'femake', 'femail', 'cis woman']

    def map_gender(g):
        if g in male_terms:
            return 'Male'
        elif g in female_terms:
            return 'Female'
        else:
            return 'Other'

    df['Gender'] = df['Gender'].apply(map_gender)

    # Handle missing values
    df = df.drop(columns=['comments'])
    df['self_employed'] = df['self_employed'].fillna(df['self_employed'].mode()[0])
    df['work_interfere'] = df['work_interfere'].fillna('Not applicable')
    df['state'] = df['state'].fillna('Unknown')

    return df

df = load_data()

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

gender_filter = st.sidebar.multiselect(
    "Gender", options=sorted(df['Gender'].unique()), default=list(df['Gender'].unique())
)

country_options = df['Country'].value_counts().head(15).index.tolist()
country_filter = st.sidebar.multiselect(
    "Country (top 15 shown)", options=country_options, default=country_options
)

age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

treatment_filter = st.sidebar.multiselect(
    "Treatment Sought", options=sorted(df['treatment'].unique()), default=list(df['treatment'].unique())
)

filtered_df = df[
    (df['Gender'].isin(gender_filter)) &
    (df['Country'].isin(country_filter)) &
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['treatment'].isin(treatment_filter))
]

# ---------------------------
# Header
# ---------------------------
st.title("🧠 Mental Health in Tech Survey — EDA Dashboard")
st.markdown(
    "Exploring attitudes towards mental health and treatment-seeking behavior "
    "among tech employees, based on the 2014 OSMI survey."
)

# ---------------------------
# Key metrics
# ---------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Respondents", len(filtered_df))
col2.metric("Avg Age", round(filtered_df['Age'].mean(), 1) if len(filtered_df) else "-")
treated_pct = (
    round((filtered_df['treatment'] == 'Yes').mean() * 100, 1) if len(filtered_df) else 0
)
col3.metric("% Sought Treatment", f"{treated_pct}%")
col4.metric("Countries Represented", filtered_df['Country'].nunique())

st.divider()

if len(filtered_df) == 0:
    st.warning("No data matches the selected filters. Please adjust the filters in the sidebar.")
    st.stop()

# ---------------------------
# Tabs for organized layout
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Demographics", "Workplace Factors", "Attitudes & Stigma", "Correlation"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df['Age'], bins=20, kde=True, color='steelblue', ax=ax)
        st.pyplot(fig)

    with c2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered_df, x='Gender',
            order=filtered_df['Gender'].value_counts().index, palette='Set2', ax=ax
        )
        st.pyplot(fig)

    st.subheader("Treatment Sought")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=filtered_df, x='treatment', palette='Set1', ax=ax)
    st.pyplot(fig)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Company Size Distribution")
        order6 = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered_df, x='no_employees',
            order=[o for o in order6 if o in filtered_df['no_employees'].unique()],
            palette='crest', ax=ax
        )
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with c2:
        st.subheader("Remote Work vs Treatment")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='remote_work', hue='treatment', palette='crest', ax=ax)
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Employer-Provided Benefits")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='benefits', palette='viridis', ax=ax)
        st.pyplot(fig)

    with c4:
        st.subheader("Awareness of Care Options")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='care_options', palette='rocket', ax=ax)
        st.pyplot(fig)

    st.subheader("Ease of Taking Medical Leave")
    order10 = ['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult']
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(
        data=filtered_df, x='leave',
        order=[o for o in order10 if o in filtered_df['leave'].unique()],
        palette='mako', ax=ax
    )
    st.pyplot(fig)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Family History vs Treatment")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='family_history', hue='treatment', palette='Set2', ax=ax)
        st.pyplot(fig)

    with c2:
        st.subheader("Work Interference Levels")
        order = ['Never', 'Rarely', 'Sometimes', 'Often', 'Not applicable']
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered_df, x='work_interfere',
            order=[o for o in order if o in filtered_df['work_interfere'].unique()],
            palette='mako', ax=ax
        )
        st.pyplot(fig)

    st.subheader("Willing to Discuss with Coworkers vs Supervisor")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=filtered_df, x='coworkers', ax=axes[0], palette='Set1')
    axes[0].set_title('Coworkers')
    sns.countplot(data=filtered_df, x='supervisor', ax=axes[1], palette='Set1')
    axes[1].set_title('Supervisor')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Perceived Negative Consequence of Discussing Mental Health")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=filtered_df, x='mental_health_consequence', palette='Set2', ax=ax)
    st.pyplot(fig)

with tab4:
    st.subheader("Correlation Heatmap (Label-Encoded Variables)")
    df_encoded = filtered_df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("Pair Plot (Key Variables)")
    cols_for_pairplot = ['Age', 'treatment', 'family_history', 'work_interfere', 'benefits']
    df_pair = df_encoded[cols_for_pairplot]
    pair_fig = sns.pairplot(df_pair, hue='treatment', palette='husl')
    st.pyplot(pair_fig)

# ---------------------------
# Raw data viewer
# ---------------------------
st.divider()
with st.expander("View Filtered Raw Data"):
    st.dataframe(filtered_df)