Mental Health in Tech Survey - EDA

Exploratory Data Analysis on the 2014 OSMI (Open Sourcing Mental Illness) survey, exploring mental health attitudes and treatment-seeking behavior among tech industry employees.

📊 Project Overview
This project analyzes 1259 survey responses across 27 features to understand:

Demographic patterns among tech employees who took the survey
Factors influencing mental health treatment-seeking behavior
Gaps in employee awareness of company mental health policies (benefits, anonymity, leave)
Workplace stigma around discussing mental health

🗂️ Files

Sample_EDA_Submission_Template.ipynb — Full EDA notebook (data wrangling, 15 visualizations, insights)
app.py — Interactive Streamlit dashboard
requirements.txt — Python dependencies
survey.csv — Raw dataset

🧹 Data Cleaning

Fixed unrealistic Age values (kept 18-75 range)
Standardized 49 inconsistent Gender entries into Male/Female/Other
Handled missing values in state, self_employed, work_interfere; dropped sparse comments column

🔍 Key Insights

Family history and work interference are stronger predictors of treatment-seeking than age or gender
Many employees are unsure (not "No") about their benefits, anonymity protection, and leave policies — indicating a communication gap
A notable share fear negative consequences from disclosing mental health issues at work

🚀 Run the Dashboard
pip install -r requirements.txt
streamlit run app.py

🛠️ Tech Stack
Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit

📌 Dataset Source
OSMI Mental Health in Tech Survey (2014) - https://www.kaggle.com/datasets/osmihelp/mental-health-in-tech-survey