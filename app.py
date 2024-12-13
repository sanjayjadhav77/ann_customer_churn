import streamlit as st
import numpy as numpy
import tensorflow as tensorflow
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

import pandas as pd
import pickle

### load the trained model ,scaler,pickle ,onehot
model=load_model("model.h5")

with open("lable_encoder_gender.pkl","rb") as file:
    lable_encoder_gender=pickle.load(file)

with open("onehot_encoder_geo.pkl","rb") as file:
    lable_encoder_geo=pickle.load(file)

with open("Scaler.pkl","rb") as file:
    scaler= pickle.load(file)

st.title("customer churn prediction")
geography =st.selectbox("geography",lable_encoder_geo.categories_[0])
gender = st.selectbox("Gender",lable_encoder_gender.classes_)
age =st.slider("Age",18,92)
balance =st.number_input("balance")
credit_score=st.number_input("Credit Score")
estimated_salary =st.number_input("Estimated  Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("number of product",1,4)
has_cr_card=st.selectbox("has credit card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])


input_data = pd.DataFrame({
"CreditScore" : [credit_score],
"Gender" : [lable_encoder_gender.transform([gender])[0]],
"Age" : [age] ,
"Tenure" :[tenure],
"Balance": [balance],
"NumOfProducts":[num_of_products],
"HasCrCard" :[has_cr_card],
"IsActiveMember" :[is_active_member],
"EstimatedSalary" :[estimated_salary]

})

geo_encoded = lable_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=lable_encoder_geo.get_feature_names_out(["Geography"]))


input_df=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled=scaler.transform(input_df)

prediction =model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write("the customer is not likely to churn.")