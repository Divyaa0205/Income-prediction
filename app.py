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
        # Extracting input data from the form
        data = {col: request.form[col] for col in numerical_cols + categorical_cols}
        print(f"Received data: {data}")

        # Convert numerical data
        numerical_data = {}
        for col in numerical_cols:
            try:
                numerical_data[col] = float(data[col])
            except ValueError:
                return jsonify({'error': f'Invalid value for {col}'}), 400

        # Prepare DataFrames for numerical and categorical data
        numerical_df = pd.DataFrame([numerical_data])  # Numerical data
        print("Numerical DataFrame:\n", numerical_df)

        categorical_data = {col: data[col] for col in categorical_cols}
        categorical_df = pd.DataFrame([categorical_data])  # Categorical data
        print("Categorical DataFrame Before Encoding:\n", categorical_df)

        # One-hot encode categorical features
        categorical_df = pd.get_dummies(categorical_df, drop_first=True)
        print("Categorical DataFrame After Encoding:\n", categorical_df)

        # Reindex categorical DataFrame to match training columns
        categorical_df = categorical_df.reindex(
            columns=[col for col in model_columns if col not in numerical_cols], fill_value=0
        )
        print("Reindexed Categorical DataFrame:\n", categorical_df)

        # Combine numerical and categorical data
        input_df = pd.concat([numerical_df, categorical_df], axis=1)
        print("Combined DataFrame Before Final Reindexing:\n", input_df)

        # Reindex to ensure input_df has the same columns as model_columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        print("Reindexed Input DataFrame:\n", input_df)

        # Scale numerical data
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        print("Final Input DataFrame After Scaling:\n", input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]
        income = ">50K" if prediction == 1 else "<=50K"

        return render_template('result.html', prediction=income)

    except Exception as e:
        print("Error encountered:", str(e))
        return jsonify({'error': str(e)}), 500




    
if __name__ == "__main__":
    app.run(debug=True)