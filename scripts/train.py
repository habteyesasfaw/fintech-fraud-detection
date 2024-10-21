# train.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import cross_val_score

def load_data(processed_fraud_path, creditcard_path):
    """ Load fraud and credit card datasets from file paths """
    processed_fraud_data = pd.read_csv(processed_fraud_path)
    creditcard_data = pd.read_csv(creditcard_path)
    return processed_fraud_data, creditcard_data


def preprocess_data(df, target_column='class'):
    # Convert date columns to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Extract relevant time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    # # Drop original date columns if not needed anymore
    # df.drop(['signup_time', 'purchase_time'], axis=1, inplace=True)

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """ Split data into training and test sets """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(model, X_train, y_train):
    """ Train a given model """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """ Evaluate model performance using test data """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

def run_baseline_models(X_train, X_test, y_train, y_test):
    """ Train and evaluate baseline models: Logistic Regression, Decision Tree, etc. """
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        trained_model = train_model(model, X_train, y_train)
        accuracy, precision, recall, f1 = evaluate_model(trained_model, X_test, y_test)
        results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
    
    return results

def build_lstm_model(input_shape):
    """ Build and compile an LSTM model """
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """ Train and evaluate the LSTM model """
    model = build_lstm_model((X_train.shape[1], 1))
    
    # Reshape data for LSTM
    X_train_lstm = np.expand_dims(X_train, axis=2)
    X_test_lstm = np.expand_dims(X_test, axis=2)
    
    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_lstm, y_test))
    
    _, test_accuracy = model.evaluate(X_test_lstm, y_test)
    print(f"LSTM Test Accuracy: {test_accuracy}")
    
    return model

def cross_validate_model(model, X, y, cv=5):
    """ Perform cross-validation on the model """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return np.mean(scores)

