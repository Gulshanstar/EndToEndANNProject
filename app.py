import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st
model = tf.keras.models.load_model('models.h5')

with open('le.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('ohe.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('mm.pkl', 'rb') as file:
    std_encoder_load = pickle.load(file)

## streamlit app
st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_features = onehot_encoder_geo.transform([[geography]])
geo_features_df = pd.DataFrame(data=geo_features, columns=['Geography_France', 'Geography_Germany', 'Geography_Spain'])


res_df = pd.concat([input_data.reset_index(drop=True), geo_features_df], axis=1)

input_scaled = std_encoder_load.transform(res_df)
pred = model.predict(input_scaled)
result = pred[0][0]
st.write(f'Churn Probability: {result:.2f}')

if(result >0.5) : 
    st.write('customer churn')
else:
    st.write('customer not churn')