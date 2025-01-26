
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def model_train():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  
    y = iris.target  

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    # Create a dictionary with dataset lengths
    dataset_lengths = {
        'train_features': len(X_train),
        'train_labels': len(y_train),
        'test_features': len(X_test),
        'test_labels': len(y_test)
    }

    print("\nDataset Lengths:")
    for key, value in dataset_lengths.items():
        print(f"{key}: {value}")
        
    return dataset_lengths

if __name__ == "__main__":
    model_train()