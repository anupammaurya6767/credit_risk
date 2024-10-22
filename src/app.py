import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression_model.pkl',
        'Decision Tree': 'models/decision_tree_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl'
    }

    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                models[model_name] = pickle.load(file)

    return models

def main():
    st.title('Credit Risk Prediction App')
    st.write('Enter the applicant\'s information to predict credit risk.')

    # Load models
    models = load_models()

    # Model selection
    model_choice = st.selectbox(
        'Select Model',
        ['Logistic Regression', 'Decision Tree', 'Random Forest']
    )

    # Create input fields
    gender = st.selectbox('Gender', ['Female', 'Male'])
    reality = st.selectbox('Owns Realty', ['No', 'Yes'])
    child_no = st.selectbox('Number of Children', ['0', '1', '2 or more'])
    work_phone = st.selectbox('Has Work Phone', ['No', 'Yes'])
    age = st.slider('Age', 18, 100, 30)
    work_experience = st.slider('Work Experience (years)', 0, 50, 5)
    occupation = st.selectbox('Occupation', ['Labor work', 'Office work', 'High-tech work'])
    family_size = st.selectbox('Family Size', ['1', '2', '3 or more'])
    housing_type = st.selectbox('Housing Type', ['Co-op apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents'])
    education = st.selectbox('Education', ['Higher education', 'Incomplete higher', 'Lower secondary', 'Secondary'])
    family_status = st.selectbox('Family Status', ['Civil marriage', 'Married', 'Separated', 'Single / not married', 'Widow'])

    # Create a dictionary to store the input values
    input_data = {
        'ID': [0],  # Adding dummy ID as it was present during training
        'Gender': [1 if gender == 'Male' else 0],
        'Reality': [1 if reality == 'Yes' else 0],
        'ChldNo_1': [1 if child_no == '1' else 0],
        'ChldNo_2More': [1 if child_no == '2 or more' else 0],
        'wkphone': [1 if work_phone == 'Yes' else 0],
        'gp_Age_high': [1 if 45 <= age < 60 else 0],
        'gp_Age_highest': [1 if age >= 60 else 0],
        'gp_Age_low': [1 if 30 <= age < 45 else 0],
        'gp_Age_lowest': [1 if age < 30 else 0],
        'gp_worktm_high': [1 if 10 <= work_experience < 20 else 0],
        'gp_worktm_highest': [1 if work_experience >= 20 else 0],
        'gp_worktm_low': [1 if 5 <= work_experience < 10 else 0],
        'gp_worktm_medium': [1 if 1 <= work_experience < 5 else 0],
        'occyp_hightecwk': [1 if occupation == 'High-tech work' else 0],
        'occyp_officewk': [1 if occupation == 'Office work' else 0],
        'famsizegp_1': [1 if family_size == '1' else 0],
        'famsizegp_3more': [1 if family_size == '3 or more' else 0],
        'houtp_Co-op apartment': [1 if housing_type == 'Co-op apartment' else 0],
        'houtp_Municipal apartment': [1 if housing_type == 'Municipal apartment' else 0],
        'houtp_Office apartment': [1 if housing_type == 'Office apartment' else 0],
        'houtp_Rented apartment': [1 if housing_type == 'Rented apartment' else 0],
        'houtp_With parents': [1 if housing_type == 'With parents' else 0],
        'edutp_Higher education': [1 if education == 'Higher education' else 0],
        'edutp_Incomplete higher': [1 if education == 'Incomplete higher' else 0],
        'edutp_Lower secondary': [1 if education == 'Lower secondary' else 0],
        'famtp_Civil marriage': [1 if family_status == 'Civil marriage' else 0],
        'famtp_Separated': [1 if family_status == 'Separated' else 0],
        'famtp_Single / not married': [1 if family_status == 'Single / not married' else 0],
        'famtp_Widow': [1 if family_status == 'Widow' else 0]
    }

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame(input_data)

    # Add a prediction button
    if st.button('Predict Credit Risk'):
        if model_choice in models:
            # Make prediction
            model = models[model_choice]
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)

            # Display the result
            st.subheader(f'Prediction Result ({model_choice})')
            if prediction[0] == 1:
                st.warning('High risk of default')
                st.write(f'Probability of default: {probability[0][1]:.2%}')
            else:
                st.success('Low risk of default')
                st.write(f'Probability of default: {probability[0][1]:.2%}')
        else:
            st.error(f"Model {model_choice} not found. Please make sure the model file exists.")

if __name__ == '__main__':
    main()
