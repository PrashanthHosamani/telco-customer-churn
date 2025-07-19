Telco Customer Churn Prediction

This project predicts customer churn for a telecom company using a trained machine learning model, and serves predictions through a Streamlit web application.

📂 Repository Structure

`├── data/

 │   └── tel_churn_clean.csv
 
 ├── models/
 │   └── model.joblib            
 ├── app.py                       
 ├── requirements.txt             
 └── README.md`                 

🚀 Getting Started

1. Clone the repository

git clone https://github.com/<your-username>/telco-customer-churn.git
cd telco-customer-churn

2. Set up Python environment

`python3 -m venv venv`

`source venv/bin/activate   # macOS/Linux`

`.\venv\Scripts\activate   # Windows`

`pip install -r requirements.txt`

3. Prepare the model

If you haven’t yet generated a trained model (models/model.joblib):

Open notebooks/Churn Analysis.ipynb.

Ensure the final cell saves your pipeline:

joblib.dump(your_trained_pipeline, 'models/model.joblib')

Run the notebook cells to produce models/model.joblib.

4. Run the Streamlit app locally

streamlit run app.py

Open http://localhost:8501 in your browser to interact with the app.

🌐 Deployment on Streamlit Cloud

Push your code to a GitHub repository.

Go to Streamlit Cloud and sign in.

Create a new app and select your GitHub repo.

Ensure the app.py path and branch are correct.

Click Deploy. Your live app URL will be provided by Streamlit.

🛠️ App Features

Single Prediction: Fill in customer details and get a churn probability & class.

Batch Prediction: Upload a CSV file, and download a file with churn predictions & probabilities for each record.

📈 Model Details

Algorithm: Gradient Boosting / XGBoost (wrapped in a Pipeline with preprocessing)

Preprocessing: Label encoding for categoricals, scaling/numerical imputation as needed.

Evaluation: Achieves ~X% accuracy / AUC on hold-out test set.

📄 References

Dataset: Telco Customer Churn

Streamlit documentation: https://docs.streamlit.io

🤝 Contributing

Feel free to open issues or submit pull requests to improve the app or model pipeline.

⚖️ License

This project is released under the MIT License.

