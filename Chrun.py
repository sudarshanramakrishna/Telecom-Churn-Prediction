import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

import seaborn as sns
from PIL import Image


 # Load data
loaded_model = pickle.load(open('/Users/shreyanthhg/Desktop/4470_108BCustomerChurninTelecomCompany/pipeline_smote.pkl', 'rb'))



@st.cache(allow_output_mutation=True)


def load_data(file):
    data = pd.read_csv(file)
    return data

# Check shape
def check_shape(data):
   
# Display image
    image = Image.open('/Users/shreyanthhg/Desktop/4470_108BCustomerChurninTelecomCompany/Capture.JPG')
    st.image(image, caption='Chrun prediction', use_column_width=True)
    st.write(data.head(5))
    if st.button('check shape'):
        st.write("Shape of the dataset:", data.shape)
    
    

# Describe
def describe_data(data):
    st.write("Descriptive statistics of the dataset:")
    st.write(data.head(5))
    if st.button('Describe'):
        st.write(data.describe())
    
def missing(data):
    count=data.isnull().sum().sort_values(ascending=False)
    perc=data.isnull().mean().sort_values(ascending=False)
    total = pd.concat([count,perc],axis=1,keys=['missing value count','missing value %'])
    st.write(data.head(5))
    if st.button('Check Missing Value'):
        st.dataframe(total)
        
        
def visualize_data(data):
    
    image = Image.open('/Users/shreyanthhg/Desktop/4470_108BCustomerChurninTelecomCompany/Capture.JPG')
    st.image(image, caption='Chrun prediction', use_column_width=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #st.beta_set_page_config(layout="wide")

    st.title("Data Visualization with Streamlit")

    # Load dataset
    #data = st.cache(pd.read_csv)("/Users/shreyanthhg/Desktop/Heart attack predication web app streamlit copy/heart_failure_clinical_records_dataset.csv")

    # Display dataset
    st.write("### Dataset")
    st.dataframe(data)

    # Uni-variate analysis
    st.sidebar.subheader("Uni-variate Analysis")
    uni_variate_cols = data.columns
    selected_uni_variate_col = st.sidebar.selectbox("Select a column", uni_variate_cols)

    if st.sidebar.button("Plot"):
        st.write("### Uni-variate Analysis")
        sns.histplot(data[selected_uni_variate_col])
        st.pyplot()

    # Bi-variate analysis
    st.sidebar.subheader("Bi-variate Analysis")
    bi_variate_cols = data.columns
    selected_bi_variate_cols = st.sidebar.multiselect("Select two columns", bi_variate_cols)

    if st.sidebar.button("Plot", key="plot_button1"):
        st.write("### Bi-variate Analysis")
        sns.jointplot(x=data[selected_bi_variate_cols[0]], y=data[selected_bi_variate_cols[1]])
        st.pyplot()

    # Multi-variate analysis
    st.sidebar.subheader("Multi-variate Analysis")
    multi_variate_cols = data.columns
    selected_multi_variate_cols = st.sidebar.multiselect("Select columns", multi_variate_cols)

    if st.sidebar.button("Plot", key="plot_button2"):
        st.write("### Multi-variate Analysis")
        sns.pairplot(data[selected_multi_variate_cols])
        st.pyplot()
        
        
                
                 
def encoding(input_data):
    label_encoder = LabelEncoder()
    object_columns = [col for col in input_data.columns if input_data[col].dtype == 'object']
    for col in object_columns:
        input_data[col] = label_encoder.fit_transform(input_data[col])
    return input_data


# Prediction
def churn_prediction(input_data, loaded_model):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    # Make predictions using the model
    prediction = loaded_model.predict_proba(input_data_reshaped)
    
    # Extract the probabilities with 2 decimal places
    positive_probability = round(prediction[:, 1][0], 2)
    negative_probability = round(prediction[:, 0][0], 2)

    # Return a dictionary containing the probabilities
    return positive_probability,negative_probability


    
# Main function
data= None

def main():
    
    st.title("Descriptive Analysis of Chrun prediction using Streamlit")
    st.sidebar.title("Descriptive Analysis")
        
    # Load data
    data = load_data("/Users/shreyanthhg/Desktop/4470_108BCustomerChurninTelecomCompany/train.csv")
    data=pd.DataFrame(data)
    # Sidebar options
    option = st.sidebar.selectbox("Select option", ["Check Shape", "Describe","Visualization", 
                                                    "Input"])
    if option == "Check Shape":
        check_shape(data)

    elif option == "Describe":
        describe_data(data)
        
        
    elif option == "Missing Value Check":
        missing(data)
        
    elif option == "Visualization":
        st.subheader("Data Visualization")
        visualize_data(data)
          
        
    elif option=='Input':
        st.title('Telecom company Customer Churn prediction')
            
            
            # getting the input data from the user
   


        # Create dropdown menus for input
        gender = st.text_input('Gender')
        SeniorCitizen = st.text_input('Senior Citizen')
        Partner = st.text_input('Partner')
        Dependents = st.text_input('Dependents')
        tenure = st.text_input('Tenure')
        PhoneService = st.text_input('Phone Service')
        MultipleLines = st.text_input('Multiple Lines')
        InternetService = st.text_input('Internet Service')
        OnlineSecurity = st.text_input('Online Security')
        OnlineBackup = st.text_input('Online Backup')
        DeviceProtection = st.text_input('Device Protection')
        TechSupport = st.text_input('Tech Support')
        StreamingTV = st.text_input('StreamingTV')
        StreamingMovies = st.text_input('StreamingMovies')
        Contract = st.text_input('Contract')
        PaperlessBilling = st.text_input('PaperlessBilling')
        PaymentMethod = st.text_input('PaymentMethod')
        MonthlyCharges = st.text_input('MonthlyCharges')
        TotalCharges = st.text_input('TotalCharges')
        
        

    
            # Create a numpy array containing the input data
        input_data=[gender, SeniorCitizen, Partner, Dependents,
                        tenure, PhoneService, MultipleLines, InternetService,
                        OnlineSecurity, OnlineBackup, DeviceProtection,TechSupport,
                        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                        PaymentMethod, MonthlyCharges, TotalCharges]
            
            
        input_data=pd.DataFrame(input_data)
        input_data=encoding(input_data)
    
       
        # code for Prediction
        chrun = ''
        
            
        if st.button('Predict'):
            positive_probability, negative_probability = churn_prediction(input_data,loaded_model)
            # Print the predictions with 2 decimal places
            st.write(positive_probability)
            st.write(negative_probability)

            # Print the predicted customer class
        if positive_probability > negative_probability:
            st.text("customer: churn")
        else:
            st.text("customer: not churn")
            
        st.success(chrun)
        
        

if __name__ == '__main__':
    main()

