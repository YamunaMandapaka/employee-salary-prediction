# employee-salary-prediction
This project aims to predict employee salaries using various machine learning models. It includes data preprocessing, model training, evaluation, and prediction steps. The application is integrated with Streamlit and Ngrok for deploying the model as a web app.

Features
Data preprocessing with pandas and scikit-learn.

Model training using:

Random Forest

Logistic Regression

Support Vector Machine (SVM)

Evaluation using metrics like Mean Squared Error and R² score.

Web-based interface using Streamlit.

Model persistence with joblib.

Ngrok tunneling for public access.

Tech Stack
Python

Pandas

Scikit-learn

Matplotlib

Streamlit

Joblib

Pyngrok

Dataset
The project uses adult.csv (stored locally in C:\Users\Yamuna\Documents\smart\adult.csv).

Predicted results are saved in predicted_classes.csv.

Project Structure
bash
Copy
Edit
.
├── employee_salary_prediction.ipynb   # Jupyter Notebook
├── adult.csv                          # Dataset
├── predicted_classes.csv              # Predictions
└── README.md                          # Project Documentation
How to Run
Clone this repository or download the files.

Install dependencies:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib streamlit joblib pyngrok
Run the Jupyter Notebook to train and evaluate the models:

bash
Copy
Edit
jupyter notebook employee_salary_prediction.ipynb
Launch the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Use Ngrok to expose the app publicly:

bash
Copy
Edit
ngrok http 8501
Results
Models were evaluated using train_test_split.

Random Forest and SVM provided better accuracy compared to Logistic Regression (details inside the notebook).

Future Improvements
Include additional features for better salary prediction.

Integrate cross-validation and hyperparameter tuning.

Deploy on a cloud platform like Heroku or AWS.

