import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import pickle
import sys
import numpy as np
#import matplotlib.pyplot as plt  # Remove Matplotlib import
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load dataset - ensure the path is correct
try:
    data = pd.read_csv('birth_weight.csv')
except FileNotFoundError:
    print("Error: Dataset file 'birth_weight.csv' not found in the current directory")
    sys.exit(1)
print(data.head(20))

# Select features and target
features = [
    'Length', 'Headcirc', 'Gestation', 'smoker', 'mage', 'mnocig',
    'mheight', 'mppwt', 'fage', 'fedyrs', 'fnocig', 'fheight', 'mage35'
]
target = 'Birthweight'

# Handle missing values in the target (y)
y = data[target].dropna()

# Handle missing values in the features (X) by imputing
X = data[features]

# Impute missing values in features (X) using mean strategy
imputer = SimpleImputer(strategy='mean')  # Can also use median
X_imputed = imputer.fit_transform(X)

# Now, ensure the target (y) aligns with the features after dropping rows with NaN
X_imputed = X_imputed[:len(y)]  # Ensure both X and y have the same number of rows

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

# Calculate MSE, RMSE, MAE, and R² for Decision Tree
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Calculate MSE, RMSE, MAE, and R² for Linear Regression
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# Output the model performance
print("Decision Tree Performance:")
print(f"MSE: {dt_mse:.2f} grams^2")
print(f"RMSE: {dt_rmse:.2f} grams")
print(f"MAE: {dt_mae:.2f} grams")
print(f"R² Score: {dt_r2:.2f}")

print("\nLinear Regression Performance:")
print(f"MSE: {lr_mse:.2f} grams^2")
print(f"RMSE: {lr_rmse:.2f} grams")
print(f"MAE: {lr_mae:.2f} grams")
print(f"R² Score: {lr_r2:.2f}")

# Confusion Matrix for Decision Tree
# Since birthweight is a continuous variable, a confusion matrix is not directly applicable.
# We need to convert the birthweight into categories (e.g., Low, Normal, High) to use a confusion matrix.

# Define birthweight categories
BIRTHWEIGHT_GUIDELINES = {
    'Low Birthweight': (0, 2500),     # Grams
    'Normal Birthweight': (2500, 4000),  # Grams
    'High Birthweight': (4000, float('inf')) # Grams
}

def categorize_birthweight(weight):
    for category, (lower, upper) in BIRTHWEIGHT_GUIDELINES.items():
        if lower <= weight <= upper:
            return category
    return "Unknown"

# Convert true and predicted birthweights to categories
y_test_categorical = y_test.apply(categorize_birthweight)
dt_pred_categorical = pd.Series([categorize_birthweight(pred) for pred in dt_pred])

# Calculate confusion matrix
cm = confusion_matrix(y_test_categorical, dt_pred_categorical, labels=['Low Birthweight', 'Normal Birthweight', 'High Birthweight'])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Normal', 'High'], yticklabels=['Low', 'Normal', 'High'])
plt.xlabel('Predicted Category')
plt.ylabel('True Category')
plt.title('Confusion Matrix - Decision Tree')
plt.show()



# Save models and scaler
try:
    with open('model/dt_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)

    with open('model/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\nModels saved successfully in 'model' directory")
except Exception as e:
    print(f"Error saving models: {e}")


# Parameter thresholds (replace with actual medical guidelines)
THRESHOLDS = {
    'Length': (40, 60),          # Length in cm (example range)
    'Headcirc': (30, 40),        # Head circumference in cm (example range)
    'Gestation': (37, 42),       # Gestation in weeks (full term range)
    'mage': (18, 45),           # Mother's age (example range)
    'mnocig': (0, 20),          # Mother's cigarettes per day (example upper limit)
    'mheight': (150, 180),       # Mother's height in cm (example range)
    'mppwt': (40, 120),        # Mother's pre-pregnancy weight in kg (example range)
    'fage': (18, 60),           # Father's age (example range)
    'fedyrs': (0, 20),          # Father's education years (example range)
    'fnocig': (0, 20),          # Father's cigarettes per day (example upper limit)
    'fheight': (160, 200),      # Father's height in cm (example range)
}



# Function to validate the input parameters against the thresholds
def validate_input(parameter, value):
    if parameter in THRESHOLDS:
        lower, upper = THRESHOLDS[parameter]
        if not (lower <= value <= upper):
            print(f"Warning: {parameter} value ({value}) is outside the recommended range ({lower}-{upper}).")
            return False  # or raise an exception if you want to enforce strict validation
    return True

# Function to get medical guidelines based on birthweight
def get_birthweight_guidelines(birthweight):
    for category, (lower, upper) in BIRTHWEIGHT_GUIDELINES.items():
        if lower <= birthweight <= upper:
            return category  # Return the category the birthweight falls into
    return "Unknown" #Should not happen, but handle just in case

# Function to predict the birthweight based on user input
def predict_birthweight():
    # Collect user input for the features
    print("\nPlease enter the following details for prediction:")
    try:
        length = float(input("Length (in cm): "))
        if not validate_input('Length', length): return
        
        headcirc = float(input("Head circumference (in cm): "))
        if not validate_input('Headcirc', headcirc): return
        
        gestation = float(input("Gestation (in weeks): "))
        if not validate_input('Gestation', gestation): return
        
        smoker = input("Smoker (Yes/No): ").strip()
        
        mage = float(input("Mother's age (in years): "))
        if not validate_input('mage', mage): return
        
        mnocig = float(input("Mother's number of cigarettes per day: "))
        if not validate_input('mnocig', mnocig): return
        
        mheight = float(input("Mother's height (in cm): "))
        if not validate_input('mheight', mheight): return
        
        mppwt = float(input("Mother's pre-pregnancy weight (in kg): "))
        if not validate_input('mppwt', mppwt): return
        
        fage = float(input("Father's age (in years): "))
        if not validate_input('fage', fage): return
        
        fedyrs = float(input("Father's education years: "))
        if not validate_input('fedyrs', fedyrs): return
        
        fnocig = float(input("Father's number of cigarettes per day: "))
        if not validate_input('fnocig', fnocig): return
        
        fheight = float(input("Father's height (in cm): "))
        if not validate_input('fheight', fheight): return
        
        mage35 = input("Is the mother aged 35 or more (Yes/No): ").strip()

        # Map categorical variables (smoker, mage35)
        smoker = 1 if smoker.lower() == 'yes' else 0
        mage35 = 1 if mage35.lower() == 'yes' else 0

        # Prepare the input for prediction
        input_data = np.array([[length, headcirc, gestation, smoker, mage, mnocig, 
                                mheight, mppwt, fage, fedyrs, fnocig, fheight, mage35]])

        # Impute missing values if any
        input_data_imputed = imputer.transform(input_data)

        # Scale the features
        input_data_scaled = scaler.transform(input_data_imputed)

        # Predict using both models
        lr_pred = lr_model.predict(input_data_scaled)
        birthweight_kg = lr_pred[0] / 1000 # Convert grams to kilograms
        
        print(f" Fetal birth weight : {birthweight_kg:.2f} Kg")

        # Get the birthweight category and associated guidelines
        birthweight_category = get_birthweight_guidelines(lr_pred[0])
        print(f"Birthweight Category: {birthweight_category}")
        # Add code here to display or use medical guidelines based on the category

    except ValueError:
        print("Invalid input. Please enter numeric values for all fields.")

# Call the prediction function
predict_birthweight()