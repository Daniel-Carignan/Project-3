import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#changing string to int 
# 1 = Under Weight
# 2 = Normal Weight
# 3 = Overweight
# 4 = Obese

#same thing here
# Male = 0
# Female = 1


# Load the data from CSV file
data = pd.read_csv('Obesity Classification.csv')



# Convert Gender column to numeric (assuming you want to include it in the model)
data['Gender'] = data['Gender'].astype('category').cat.codes

# Split the dataset into features (X) and target variable (y)
X = data[['Age', 'Gender', 'Height', 'Weight', 'BMI']]  # Features
y = data['Label']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression object
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients
print('Coefficients:', model.coef_)

# Print the intercept
print('Intercept:', model.intercept_)




# Extracting features and target variable
X = data[['Age']].values  # Feature: Age
y = data['Label'].values  # Target variable

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plotting the data and the linear regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=3, label='Linear Regression')
plt.xlabel('Age')
plt.ylabel('Label')
plt.title('Linear Regression: Age vs Label')
plt.legend()
plt.show()



################################################linear regression ^^



####Logistic regression model
# Mapping labels to binary values: 1 if obese 
# (3 or 4 which is obese of over weight), 0 otherwise(categories 1 and 2)
data['Obese'] = (data['Label'].isin([3, 4])).astype(int)

# Extracting features and target variable
X = data.drop(['Label', 'Obese'], axis=1).values  # Features (excluding 'Label' and 'Obese')
y = data['Obese'].values  # Target variable: 1 if Obese, 0 otherwise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))




####pie chart for logistic regression for all the labels and all the patients

# Mapping labels to their corresponding categories
label_mapping = {1: 'Under weight', 2: 'Normal weight', 3: 'Over weight', 4: 'Obese'}
data['Label_Category'] = data['Label'].map(label_mapping)

# Count the occurrences of each label category
label_counts = data['Label_Category'].value_counts()

# Plotting the pie chart
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of People in Each Label Category')
plt.show()




#### chart for finding correlation between age and obesity 


# # Map label values to descriptions
label_mapping = {1: 'Underweight', 2: 'Normal weight', 3: 'Overweight', 4: 'Obese'}

plt.legend(["1: Underweight", "2: Normal weight", "3: Overweight", "4: Obese"], loc="upper right")


# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the bar chart
data.plot(kind='bar', x='Age', y='Label', ax=ax, legend=False)

# Customize y-axis ticks
plt.yticks([1, 2, 3, 4])


# Customize legend
legend_labels = [f"{key}: {value}" for key, value in label_mapping.items()]
ax.legend(legend_labels, title='Label')

plt.xlabel('Age')
plt.ylabel('Label')
plt.title('Ages with Custom Labels')
plt.show()


################################################logistic regression ^^