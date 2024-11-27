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

    cat_df = cat_df.reindex(columns=['workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay', 'marital_status_Divorced', 'marital_status_Married-AF-spouse', 'marital_status_Married-civ-spouse', 'marital_status_Married-spouse-absent', 'marital_status_Never-married', 'marital_status_Separated', 'marital_status_Widowed', 'occupation_Adm-clerical', 'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving', 'relationship_Husband', 'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'sex_Female', 'sex_Male'], fill_value=0)

    # mismatch_found_1 = False
    # for i, (input_col, model_col) in enumerate(zip(cat_df.columns, model_columns)):
    #     if input_col != model_col:
    #         print(f"Column mismatch at position {i}: Input column '{input_col}' does not match model column '{model_col}'.")
    #         mismatch_found_1 = True

    # if not mismatch_found_1:
    #     print("ALL columns matched 1")

    numerical_data = {col: int(form_data[col]) for col in numerical_cols}
    num_df = pd.DataFrame([numerical_data])

   
    input_df = pd.concat([num_df, cat_df], axis=1)

    print("before scaling:", input_df.columns)

    mismatch_found_2 =False
    for i, (input_col, model_col) in enumerate(zip(input_df.columns, model_columns)): # replaced the cat_df by input_df
        if input_col != model_col:
            print(f"Column mismatch at position {i}: Input column '{input_col}' does not match model column '{model_col}'.")
            mismatch_found_2 = True

    if not mismatch_found_2:
        print("ALL columns matched 2")

 
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    #input_df = input_df.drop(columns=numerical_cols)
    print(input_df.columns)
    
    input_df = input_df.fillna(0)

    print("after scaling and preprocessing:", input_df.columns)
   
    prediction = model.predict(input_df)[0]
    income_prediction = ">50K" if prediction == 1 else "<=50K"

    return render_template('result.html', prediction=income_prediction)

if __name__ == "__main__":
    app.run(debug=True)
