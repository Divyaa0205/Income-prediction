from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

numerical_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
categorical_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = {key: request.form.get(key) for key in ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex']}

    cat_data = {col: form_data[col] for col in categorical_cols}
    cat_df = pd.DataFrame([cat_data])
    cat_df = pd.get_dummies(cat_df, columns=categorical_cols, drop_first=False)
    # print(cat_df)

    cat_df = cat_df.reindex(columns=model_columns, fill_value=0)
    # print(cat_df)
    # print("Input Data Columns:", cat_df.columns)
    # print("Model Columns:", model_columns)
    print(set(cat_df.columns) - set(model_columns))

    numerical_data = {col: int(form_data[col]) for col in numerical_cols}
    # print(numerical_data)
    num_df = pd.DataFrame([numerical_data])

    input_df = pd.concat([num_df, cat_df], axis=1)
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    input_df = input_df.drop(columns=numerical_cols)
    # print(input_df.columns)
    input_df = input_df.fillna(0)

    print(input_df)
    prediction = model.predict(input_df)[0]
    income_prediction = ">50K" if prediction == 1 else "<=50K"

    return render_template('result.html', prediction=income_prediction)



if __name__ == "__main__":
    app.run(debug=True)
