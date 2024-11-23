from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)


# Loading the trained model

with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

numerical_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = {col: request.form[col] for col in numerical_cols + categorical_cols}

        
        numerical_data = {}
        for col in numerical_cols:
            try:
                numerical_data[col] = float(data[col])
            except ValueError:
                return jsonify({'error': f'Invalid value for {col}'}), 400

        categorical_data = {col: data[col] for col in categorical_cols}

        
        numerical_df = pd.DataFrame([numerical_data])

        
        categorical_df = pd.DataFrame([categorical_data])
        categorical_df = pd.get_dummies(categorical_df, drop_first=True)

    
        for col in model_columns:
            if col not in categorical_df.columns and col not in numerical_df.columns:
                categorical_df[col] = 0

        
        input_df = pd.concat([numerical_df, categorical_df], axis=1)
        input_df = input_df[model_columns]

        
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Predict income
        prediction = model.predict(input_df)[0]
        income = ">50K" if prediction == 1 else "<=50K"

        return render_template('result.html',prediction = income)

    except Exception as e:
        return jsonify({'error': str(e)})
   
    
if __name__ == "__main__":
    app.run(debug=True)