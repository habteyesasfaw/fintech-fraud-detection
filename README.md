
# Fraud Detection Project

## Overview
This project aims to enhance the detection of fraud cases in e-commerce and banking transactions. By leveraging advanced machine learning techniques and detailed data analysis, we aim to build a robust fraud detection system that improves transaction security and builds trust with customers and financial institutions.

## Table of Contents
- [Business Need](#business-need)
- [Datasets](#datasets)
- [Tasks](#tasks)
  - [Task 1 - Data Analysis and Preprocessing](#task-1---data-analysis-and-preprocessing)
  - [Task 2 - Model Building and Training](#task-2---model-building-and-training)
  - [Task 3 - Model Explainability](#task-3---model-explainability)
  - [Task 4 - Model Deployment and API Development](#task-4---model-deployment-and-api-development)
  - [Task 5 - Build a Dashboard with Flask and Dash](#task-5---build-a-dashboard-with-flask-and-dash)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Business Need
Adey Innovations Inc. focuses on creating solutions for e-commerce and banking. The need for an efficient fraud detection system is paramount to prevent financial losses and maintain customer trust.

## Datasets
- **Fraud_Data.csv**: Contains e-commerce transaction data with features like user ID, purchase time, purchase value, and more.
- **IpAddress_to_Country.csv**: Maps IP addresses to countries.
- **creditcard.csv**: Contains bank transaction data with anonymized features and transaction amounts.

## Tasks

### Task 1 - Data Analysis and Preprocessing
1. **Handle Missing Values**: Impute or drop missing values.
2. **Data Cleaning**: Remove duplicates and correct data types.
3. **Exploratory Data Analysis (EDA)**:
   - Univariate and bivariate analysis to understand data distributions and relationships.
4. **Merge Datasets for Geolocation Analysis**: Convert IP addresses to integer format and merge with `IpAddress_to_Country.csv`.
5. **Feature Engineering**:
   - Calculate transaction frequency and velocity.
   - Create time-based features (hour_of_day, day_of_week).
6. **Normalization and Scaling**: Standardize features for modeling.
7. **Encode Categorical Features**: Convert categorical variables into numerical format.

### Task 2 - Model Building and Training
1. **Data Preparation**: Separate features and targets and perform a train-test split.
2. **Model Selection**: Compare various models including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Multi-Layer Perceptron (MLP)
   - Convolutional Neural Network (CNN)
   - Recurrent Neural Network (RNN)
   - Long Short-Term Memory (LSTM)
3. **Model Training and Evaluation**: Train models on both datasets and evaluate their performance.
4. **MLOps Steps**: Implement versioning and experiment tracking using tools like MLflow.

### Task 3 - Model Explainability
1. **Using SHAP for Explainability**: Install SHAP and use it to explain model predictions.
   - Generate summary plots, force plots, and dependence plots.
2. **Using LIME for Explainability**: Install LIME and explain individual predictions.
   - Create feature importance plots.

### Task 4 - Model Deployment and API Development
1. **Setting Up the Flask API**: Create the Flask application and define API endpoints.
   - Implement error handling and logging.
2. **Dockerizing the Flask Application**: Create a Dockerfile to containerize the application.
3. **Build and Run the Docker Container**: Build and run the Docker image for deployment.
4. **Logging**: Integrate Flask-Logging to monitor requests, errors, and predictions.

### Task 5 - Build a Dashboard with Flask and Dash
1. **Create Flask Endpoints**: Implement endpoints to serve summary statistics and fraud trends.
2. **Set Up Dash Application**: Create an interactive dashboard using Dash to visualize insights.
3. **Dashboard Insights**:
   - Display total transactions, fraud cases, and fraud percentages.
   - Create a line chart showing detected fraud cases over time.
   - Analyze geographical distribution of fraud cases.
   - Compare fraud cases across different devices and browsers.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud_detection_project.git
   ```
2. Change into the project directory:
   ```bash
   cd fraud_detection_project
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask API:
   ```bash
   python serve_model.py
   ```
2. Access the API endpoints using Postman or curl.
3. Start the Dash dashboard:
   ```bash
   python dash_app.py
   ```
4. Open a web browser and navigate to `http://127.0.0.1:8050` to view the dashboard.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
```
