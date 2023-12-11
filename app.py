from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

model = joblib.load('trained_model.joblib')

le = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

# loan approval predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {
            'Gender': request.form['Gender'],
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Property_Area': request.form['Property_Area'],
        }
 # dataframe for input data
        input_df = pd.DataFrame([input_data])

        # categorical variables to numerical 
        for column in input_df.select_dtypes(include=['object']).columns:
            input_df[column] = le.fit_transform(input_df[column])

        # loan approval predictions
        prediction_proba = model.predict_proba(input_df)
        prediction = model.predict(input_df)

        # Approved or not approved
        loan_approval = "Approved" if prediction[0] == 1 else "Not Approved"

        # reason
        reasons = {'Risk Chance': prediction_proba[0][0]}

        return render_template('result.html', loan_approval=loan_approval, reasons=reasons)

if __name__ == '__main__':
    app.run(debug=True)


