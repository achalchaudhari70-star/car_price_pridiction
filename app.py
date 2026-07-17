import streamlit as st
import pandas as pd
import joblib
# Observations:
# 1. Streamlit helps build an interactive web application.
# 2. Pandas is used for data handling and manipulation.
# 3. Joblib loads the saved model and preprocessing files
# Load the trained Linear Regression model
model = joblib.load("LR_ford_car.pkl")

# Load the StandardScaler object
scaler = joblib.load("scaler.pkl")
# Load the encoded column namesS
encoded_columns = joblib.load("columns.pkl")

# Display success message
print("Model, Scaler, and Encoded Columns loaded successfully.")
# Configure the Streamlit page
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)
# The page title is shown in the browser tab.
# The "centered" layout keeps the content in the middle,
# making the application simple and easy to read.
# Display the main title
st.title("Ford Car Price Predictor")
# Display a short description
st.write("Enter the car details below to predict its selling price.")
# The title tells the user what the application does.
# The description provides simple instructions before entering the details.
# Numerical Input Fields

year = st.number_input(
    "Manufacturing Year",
    min_value=1996,
    max_value=2025,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=200000,
    value=20000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    max_value=600,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.0,
    max_value=5.0,
    value=1.5
)
# Comments:
# 1. st.number_input() allows the user to enter numeric values.
# 2. min_value and max_value restrict the input range.
# 3. value sets the default value displayed in the input field.
# Dropdown for Transmission
transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

# Dropdown for Fuel Type
fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Other"]
)

# Comments:
# 1. st.selectbox() displays a dropdown list of options.
# 2. It allows the user to select only one value.
# 3. It helps prevent invalid input and makes the application easy to use.
# Text input for car model
car_name = st.text_input("Enter Car Model")


# Predict button
predict = st.button("Predict Price")

# Comments:
# 1. st.text_input() allows the user to enter the car model name.
# 2. st.button() creates a button that starts the prediction process.
# 3. The prediction will be performed only when the button is clicked.
# Predict Price
if predict:

    # Create DataFrame from user input
    input_data = pd.DataFrame({
        "model": [car_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })


    # Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)

    # Match input columns with training columns
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Scale numerical features
    numeric_columns = ["year", "mileage", "tax", "mpg", "engineSize"]
    input_encoded[numeric_columns] = scaler.transform(input_encoded[numeric_columns])

    # Predict car price
    predicted_price = model.predict(input_encoded)

    # Display prediction
    st.success(f"Predicted Car Price: £{predicted_price[0]:,.2f}")
# Comments:
# 1. User input is converted into a DataFrame.
# 2. One-Hot Encoding converts categorical values into numeric values.
# 3. reindex() ensures the input has the same columns as the training data.
# 4. Numeric features are scaled using the saved StandardScaler.
# 5. The trained Linear Regression model predicts the car price.
# 6. The predicted price is displayed with the £ symbol and two decimal places.
