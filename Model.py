import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset

# Specify the path to your CSV file using a raw string
csv_file_path = r"C:\Users\Samson\Desktop\Heart prediction\Heart_Disease_Prediction.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

df.columns

# Define features (X) and target (y)
# Replace 'target' with the actual target column name after inspecting the column names
X = df.drop(columns=['Heart Disease'])  # Replace 'target' with the actual target column name
y = df['Heart Disease']  # Replace 'target' with the actual target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
print("\nLogistic Regression:")
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_predictions = log_reg.predict(X_test)
print(classification_report(y_test, log_reg_predictions))

# Random Forest
print("\nRandom Forest Classifier:")
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
rf_predictions = rf_clf.predict(X_test)
print(classification_report(y_test, rf_predictions))

# Support Vector Machine
print("\nSupport Vector Machine:")
svc_clf = SVC()
svc_clf.fit(X_train, y_train)
svc_predictions = svc_clf.predict(X_test)
print(classification_report(y_test, svc_predictions))

# Compare Accuracy Scores
print("\nAccuracy Scores:")
print("Logistic Regression:", accuracy_score(y_test, log_reg_predictions))
print("Random Forest:", accuracy_score(y_test, rf_predictions))
print("Support Vector Machine:", accuracy_score(y_test, svc_predictions))
