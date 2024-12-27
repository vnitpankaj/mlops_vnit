import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',  # multiclass classification
    num_class=3,               # number of classes
    learning_rate=0.1,         # learning rate
    max_depth=3,               # maximum depth of trees
    n_estimators=100,          # number of trees
    random_state=42
)

# Train the model
xgb_clf.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    verbose=True
)

# Make predictions
y_pred = xgb_clf.predict(X_test)

# Print evaluation metrics
print("\nModel Evaluation:")
print("----------------")
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Perform k-fold cross-validation
cv_scores = cross_val_score(xgb_clf, X, y, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Average CV score:", cv_scores.mean())
print("CV score standard deviation:", cv_scores.std())

# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_clf)
plt.title('Feature Importance in XGBoost Model')
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Function to make new predictions
def predict_iris_class(features):
    """
    Make predictions for new iris flowers
    
    Parameters:
    features (array-like): Array containing the four features in order:
                          sepal length, sepal width, petal length, petal width
    """
    features = np.array(features).reshape(1, -1)
    prediction = xgb_clf.predict(features)
    return iris.target_names[prediction[0]]

# Example usage of prediction function
sample_flower = [5.1, 3.5, 1.4, 0.2]  # Example measurements
print("\nPrediction for sample flower:", predict_iris_class(sample_flower))