#import necesary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
import plotly.express as px

#create the app title
st.title ("LOAN DEFAULT APP")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Homepage", "Data Information", "Visualization", "Machine Learning Model"])

# Load the dataset
df=pd.read_csv ("C:\\Users\\Administrator\\OneDrive\\Desktop\\Loan default\\Loan_Default.csv")

# Homepage
if option == "Homepage":
    st.title('HOMEPAGE')
    st.write('''Welcome to loan default  app homepage
             
    Financial institutions face significant challenges in assessing the creditworthiness of loan applicants. 
    Accurate evaluation of an applicant's ability to repay a loan is crucial for minimising defaults and managing risk. 
    Traditional methods of credit assessment may not fully capture the complexities of individual financial behaviour, 
    leading to potential losses for lenders. By leveraging machine learning and predictive analytics, 
    institutions can enhance their loan approval processes and make more informed decisions.
       ''')
   
   # Data Information
elif option == "Data Information":
    st.write("### Data Information")
    
    # Display the first 5 rows
    st.write('The first five rows of the dataset:', df.head())
    
    # User input: number of rows to display
    num_rows = st.slider("Select the number of rows", min_value=1, max_value=len(df), value=5)
    st.write("Here are the rows you have selected from the dataset:")
    st.write(df.head(num_rows))
    
    # Display the summary statistics of the dataset
    st.write('Summary statistics of the dataset:')
    st.write(df.describe())
    
    # Check for duplicates
    if st.checkbox("Check for duplicates"):
        st.write(df[df.duplicated()])
    
    # Total number of duplicates
    if st.checkbox("Check for total number of duplicates"):
        st.write(df.duplicated().sum())
    
    #display the number of rows and columns
    df.shape
    st.write('number of rows:',df.shape[0])
    st.write('number of columns:',df.shape[1])

    # Check for missing values
    if st.checkbox("Check for missing values in each column"):
        st.write(df.isnull().sum())

    # Total number of missing values
    if st.checkbox("Check for total number of missing values"):
        st.write(df.isnull().sum().sum())

    #clean nulls 
    #replace null values with mean in float data type
    df['rate_of_interest'].fillna(df['rate_of_interest'].mean(),inplace=True)
    df['term'].fillna(df['term'].mean(),inplace=True)
    df['Upfront_charges'].fillna(df['Upfront_charges'].mean(),inplace=True)
    df['property_value'].fillna(df['property_value'].mean(),inplace=True)
    df['income'].fillna(df['income'].mean(),inplace=True)
    df['LTV'].fillna(df['LTV'].mean(),inplace=True)
    df['dtir1'].fillna(df['dtir1'].mean(),inplace=True)
    df['Interest_rate_spread'].fillna(df['Interest_rate_spread'].mean(),inplace=True)

#replace objects or categories and objects  with mode
    df['approv_in_adv'].fillna(df['approv_in_adv'].mode()[0],inplace=True)
    df['loan_limit'].fillna(df['loan_limit'].mode()[0],inplace=True)
    df['loan_purpose'].fillna(df['loan_purpose'].mode()[0],inplace=True)
    df['Neg_ammortization'].fillna(df['Neg_ammortization'].mode()[0],inplace=True)
    df['age'].fillna(df['age'].mode()[0],inplace=True)
    df['submission_of_application'].fillna(df['submission_of_application'].mode()[0],inplace=True)

    if st.checkbox("check the total number of missing values after cleaning"):
        st.write(df.isnull().sum().sum())

        #checking for categorical and non categorical columns
    categorical_column=[]
    non_categorical_column=[]
    for column in df.columns:
      if df[column].dtype =='object' or df[column].dtype =='category':
        categorical_column.append(column)
      else:
        non_categorical_column.append(column)


    
elif option =="Visualization":
    st.write("##DATA VISULIZATION")

    # Count plot for Loan Status
    st.write("Loan Status Distribution")
    status_count = df['Status'].value_counts()
    fig1 = plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Status', palette='Set1')
    plt.title("Loan Status Distribution")
    st.pyplot(fig1)
    st.write("Interpretation: This plot shows the number of default vs. non-default loans.")

    # Pie chart for Gender distribution
    st.write("Gender Distribution")
    gender_count = df['Gender'].value_counts()
    fig2 = plt.figure(figsize=(6,4))
    plt.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title("Gender Distribution")
    st.pyplot(fig2)
    st.write("Interpretation: This pie chart visualizes the percentage of male and female borrowers.")
    
    # Boxplot of Loan Amount by Status
    st.write("Loan Amount by Loan Status")
    fig3 = plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x='Status', y='loan_amount', palette='Set2')
    plt.title("Loan Amount by Loan Status")
    st.pyplot(fig3)
    st.write("Interpretation: This plot shows the distribution of loan amounts for defaulted and non-defaulted loans.")

    # Histogram for Rate of Interest distribution
    st.write("Rate of Interest Distribution")
    fig4 = plt.figure(figsize=(6,4))
    sns.histplot(df['rate_of_interest'], bins=20, kde=True, color='green')
    plt.title("Rate of Interest Distribution")
    st.pyplot(fig4)
    st.write("Interpretation: This histogram visualizes how the rate of interest is distributed across loans.")
    
    # Heatmap for correlation
    # Select only non-categorical columns (numerical columns)
    non_categorical_column = df.select_dtypes(exclude=['object', 'category'])

    # Compute the correlation matrix of non-categorical columns
    correlation_matrix = non_categorical_column.corr()

     # Plot the correlation matrix using a heatmap
    st.write("Correlation Heatmap of Non-Categorical Columns")
    fig6 = plt.figure(figsize=(16,14))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap of Non-Categorical Columns")
    st.pyplot(fig6)
    st.write("Interpretation: The heatmap helps identify correlations between different features, highlighting key relationships.")

    
    # Label Encoding for categorical_column
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object','category']).columns #select columns with object or category types
    for column in categorical_columns:
       df[column] = label_encoder.fit_transform(df[column])
    #Setting th style for the plots
    sns.set(rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":25,"xtick.labelsize":25,"ytick.labelsize":25,
            "legend.fontsize":15})

#Creating a figure for the plots
    fig,ax=plt.subplots(4,3, figsize=(40,30))
    fig.suptitle('Distribution of the Numerical Columns data types', fontsize=40,color='green')

#Plotting the Distribution plots of the Numerical columns
    

    sns.distplot(df['loan_limit'],ax=ax[0,1],kde=True,bins=20)
    ax[0,1].set_title('loan_limit')
 
    sns.distplot(df['Gender'],ax=ax[0,2],kde=True,bins=20)
    ax[0,2].set_title('Gender')

    

    sns.distplot(df['loan_type'],ax=ax[1,0],kde=True,bins=20)
    ax[1,0].set_title('loan_type')

    sns.distplot(df['loan_purpose'],ax=ax[1,1],kde=True,bins=20)
    ax[1,1].set_title('loan_purpose')

    sns.distplot(df['Credit_Worthiness'],ax=ax[1,2],kde=True,bins=20)
    ax[1,2].set_title('Credit_Worthiness')

    sns.distplot(df['open_credit'],ax=ax[2,0],kde=True,bins=20)
    ax[2,0].set_title('open_credit')

    sns.distplot(df['business_or_commercial'],ax=ax[2,1],kde=True,bins=20)
    ax[2,1].set_title('business_or_commercial')

    sns.distplot(df['interest_only'],ax=ax[2,2],kde=True,bins=20)
    ax[2,2].set_title('interest_only')

    sns.distplot(df['lump_sum_payment'],ax=ax[3,0],kde=True,bins=20)
    ax[3,0].set_title('lump_sum_payment')

    sns.distplot(df['construction_type'],ax=ax[3,1],kde=True,bins=20)
    ax[3,1].set_title('construction_type')

    sns.distplot(df['occupancy_type'],ax=ax[3,2],kde=True,bins=20)
    ax[3,2].set_title('occupancy_type')

    sns.distplot(df['Secured_by'],ax=ax[0,0],kde=True,bins=20)
    ax[0,0].set_title('Secured_by')

    sns.distplot(df['total_units'],ax=ax[0,1],kde=True,bins=20)
    ax[0,1].set_title('total_units')

    sns.distplot(df['credit_type'],ax=ax[1,0],kde=True,bins=20)
    ax[1,0].set_title('credit_type')

    sns.distplot(df['co-applicant_credit_type'],ax=ax[1,1],kde=True,bins=20)
    ax[1,1].set_title('co-applicant_credit_type')

    

    sns.distplot(df['submission_of_application'],ax=ax[2,0],kde=True,bins=20)
    ax[2,0].set_title('submission_of_application')

    sns.distplot(df['Region'],ax=ax[2,1],kde=True,bins=20)
    ax[2,1].set_title('Region')

    sns.distplot(df['Security_Type'],ax=ax[2,2],kde=True,bins=20)
    ax[2,2].set_title('Security_Type')

#Adjust the layout
    fig.tight_layout(pad=4.0)
    fig.subplots_adjust(top=0.93)

  # Bar
    age_group = df.groupby('age').count()['Status'].sort_values(ascending = True)
    x = age_group.index
    y = age_group.values
    plt.bar(x, y)
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    st.pyplot(fig)

  # Scatter plot
    sns.scatterplot(data=df, x='loan_amount', y='Credit_Score', hue='Status', )
    plt.title('Loan Amount vs Credit Score by Default Status')
    plt.xlabel('Loan Amount')
    plt.ylabel('Credit Score')
    st.pyplot(fig)






    
elif option == "Machine Learning Model":
    st.write("WELCOME TO LOAN DEFAULT PREDIcTION ")

    #drop unnncesseary columns 
    df.drop('ID',axis=1,inplace=True)

    # Label Encoding for categorical_column
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object','category']).columns #select columns with object or category types
    for column in categorical_columns:
       df[column] = label_encoder.fit_transform(df[column])
    st.write(df.head())
    #feature engineering
#separate the feature and target variable
    x=df.drop(columns='Status',axis =1)
    y=df['Status']

    #split the dataset into train and test
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    from sklearn.metrics import accuracy_score, classification_report

# Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
    model.fit(x_train, y_train)

# Make predictions on the test set
    y_pred = model.predict(x_test)

# model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, zero_division=1)

    st.write('Accuracy:', accuracy)
    st.write('Classification Report:\n', classification_rep)

    # Fit the RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(x, y)

# Get feature importances
    importances = model.feature_importances_
    features = x.columns

# Create a DataFrame to hold the features and their importance
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot using Seaborn
    st.write("Feature Importances")
    fig, ax = plt.subplots(figsize=(14,10))  # Create a figure for the plot
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    ax.set_title('Feature Importances')

# Display the plot in Streamlit
    st.pyplot(fig)


# Define input fields for user input
    gender = st.selectbox('Gender', ['Male', 'Female'])
    loan_amount = st.number_input('Loan Amount', min_value=0)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900)
    rate_of_interest = st.number_input('Rate of Interest', min_value=0.0, max_value=50.0, step=0.01)
    term = st.number_input('Loan Term (in months)', min_value=12, max_value=360)

# Button for prediction
    if st.button('Predict'):
        # Create a new DataFrame to hold the user's input
        input_data = pd.DataFrame({
         'Gender': [gender],
         'loan_amount': [loan_amount],
         'Credit_Score': [credit_score],
         'rate_of_interest': [rate_of_interest],
         'term': [term],
         'year': [2024],  
         'loan_limit': ['N'],  
         'approv_in_adv': ['N'], 
         'loan_type': ['Type1'], 
         'loan_purpose': ['Purpose1'],  
         'open_credit': [0],  
         'business_or_commercial': ['No'],  
         'Upfront_charges': [100], 
         'interest_only': ['No'],  
         'lump_sum_payment': ['No'],  
         'property_value': [200000],  
         'construction_type': ['Type1'],  
         'occupancy_type': ['Owner'],  
         'Secured_by': ['Yes'],  
         'total_units': [1],  
         'income': [50000],  
         'credit_type': ['Type1'],  
         'co-applicant_credit_type': ['Type1'],  
         'age': [35],  
         'submission_of_application': ['Yes'],  
         'Security_Type': ['Type1'],  
         'dtir1': [30],  
         'Credit_Worthiness': ['Good'],  # Add missing feature with placeholder
         'Interest_rate_spread': [0.5],  # Add missing feature with placeholder
         'Neg_ammortization': ['No'],    # Add missing feature with placeholder
         'LTV': [80],                    # Add missing feature with placeholder
         'Region': ['East']
    })

    
    # Map Gender to numerical values (assuming the model was trained on numeric values)
        input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0})
        input_data['loan_limit'] = input_data['loan_limit'].map({'N': 0, 'Y': 1})
        input_data['approv_in_adv'] = input_data['approv_in_adv'].map({'N': 0, 'Y': 1})
        input_data['business_or_commercial'] = input_data['business_or_commercial'].map({'No': 0, 'Yes': 1})
        input_data['interest_only'] = input_data['interest_only'].map({'No': 0, 'Yes': 1})
        input_data['lump_sum_payment'] = input_data['lump_sum_payment'].map({'No': 0, 'Yes': 1})
        input_data['construction_type'] = input_data['construction_type'].map({'Type1': 1, 'Type2': 2})
        input_data['occupancy_type'] = input_data['occupancy_type'].map({'Owner': 1, 'Renter': 0})
        input_data['Secured_by'] = input_data['Secured_by'].map({'Yes': 1, 'No': 0})
        input_data['credit_type'] = input_data['credit_type'].map({'Type1': 1, 'Type2': 2})
        input_data['co-applicant_credit_type'] = input_data['co-applicant_credit_type'].map({'Type1': 1, 'Type2': 2})
        input_data['submission_of_application'] = input_data['submission_of_application'].map({'Yes': 1, 'No': 0})
        input_data['Security_Type'] = input_data['Security_Type'].map({'Type1': 1, 'Type2': 2})
        input_data['loan_type'] = input_data['loan_type'].map({'Type1': 1, 'Type2': 2})  # Correct encoding
        input_data['loan_purpose'] = input_data['loan_purpose'].map({'Purpose1': 1, 'Purpose2': 2})  # Adjust as needed
        input_data['Credit_Worthiness'] = input_data['Credit_Worthiness'].map({'Good': 1, 'Bad': 0})  # Example mapping
        input_data['Neg_ammortization'] = input_data['Neg_ammortization'].map({'No': 0, 'Yes': 1})
        input_data['Region'] = input_data['Region'].map({'East': 1, 'West': 2, 'North': 3, 'South': 4})  # Adjust as per your data


    

    # Ensure input_data columns match the trained model’s feature names
        input_data = input_data[model.feature_names_in_]

    # Make a prediction
        prediction = model.predict(input_data)[0]

    # Output the result
        if prediction == 1:
          st.write("Prediction: Loan will likely default.")
        else:
          st.write("Prediction: Loan will likely not default.")








             

