     Expresso Churn Prediction App



The Expresso Churn Prediction App is a machine learning project developed as part of a data science checkpoint.
It predicts whether a telecom customer is likely to churn (leave the network) based on behavioral and usage data from Expresso Telecom, which operates in Mauritania and Senegal.

The model analyzes customer metrics such as recharge frequency, data usage, and revenue to help the company take proactive retention actions.


    Project Objective


To build and deploy a Streamlit web application that:

Takes user input for key customer features

Uses a trained ML model to predict churn probability

Displays results in a clean, interactive dashboard



    Dataset Description

The dataset used in this project comes from the Expresso Churn Prediction Challenge on the Zindi platform.
It contains 2.5 million client records and over 15 behavioral variables such as:
| Feature              | Description                               |
| -------------------- | ----------------------------------------- |
| REGION               | Customer location                         |
| MONTANT              | Recharge amount                           |
| FREQUENCE_RECH       | Frequency of recharge                     |
| REVENUE              | Total revenue generated                   |
| ARPU_SEGMENT         | Average revenue per user                  |
| FREQUENCE            | Frequency of usage                        |
| DATA_VOLUME          | Internet data consumption                 |
| ON_NET, ORANGE, TIGO | Network usage metrics                     |
| ZONE1, ZONE2         | Service area indicators                   |
| MRG                  | Marital group                             |
| FREQ_TOP_PACK        | Top-up frequency                          |
| TENURE               | Customer duration                         |
| CHURN                | Target variable (0 = No churn, 1 = Churn) |



    Machine Learning Pipeline

Data Cleaning – handled missing values, duplicates, and outliers

Feature Encoding – categorical variables were label-encoded

Feature Scaling – numeric features normalized using StandardScaler

Model Training – Logistic Regression was used for prediction

Model Serialization – Model saved as expresso_churn_model.pkl

Deployment – Streamlit app for real-time churn prediction



      Streamlit App Features

1. User-friendly sidebar input form

2.  Instant churn prediction output

3. Display of dataset preview

4. Clean layout and responsive UI




       Tech Stack

Python

Streamlit

Pandas / NumPy

Scikit-learn

Joblib







      How to Run Locally

Clone this repository:git clone https://github.com/dupsyjj/Express_churn_prediction_app.git




Navigate into the project directory:cd Express_churn_prediction_app



Install dependencies:pip install -r requirements.txt




Run the Streamlit app:streamlit run app.py





       App Preview

Here’s a screenshot of the working app interface:








