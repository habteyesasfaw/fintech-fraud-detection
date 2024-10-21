

# Fraud Detection - Task 1: Data Analysis and Preprocessing

This repository contains the code and documentation for **Task 1** of the fraud detection project at Adey Innovations Inc. The task involves preparing and preprocessing the transaction data to improve fraud detection accuracy.

## Project Overview

The goal of this project is to detect fraudulent transactions for both e-commerce and bank credit card transactions. In **Task 1**, we focus on cleaning and analyzing the data to ensure it's ready for model building.

### Key Steps in Task 1

1. **Handling Missing Values**  
   - Impute missing values where necessary or drop rows with missing information.
  
2. **Data Cleaning**  
   - Remove duplicates.  
   - Correct data types for accurate processing.

3. **Exploratory Data Analysis (EDA)**  
   - **Univariate Analysis**: Analyze each feature individually (distribution of age, purchase value, etc.).  
   - **Bivariate Analysis**: Examine the relationships between pairs of variables (e.g., purchase value vs. fraud status).

4. **Geolocation Analysis**  
   - Merge the transaction data (`Fraud_Data.csv`) with the IP address mapping (`IpAddress_to_Country.csv`) for geolocation-based insights.

5. **Feature Engineering**  
   - Create new features like:
     - **Transaction frequency**: How often a user makes purchases.
     - **Transaction velocity**: How quickly a user makes consecutive purchases.
     - **Time-based features**: Including the hour of the day and day of the week for each transaction.

6. **Normalization and Scaling**  
   - Normalize numerical features to improve model performance.
  
7. **Encoding Categorical Features**  
   - Convert categorical variables (e.g., browser, source) into numerical formats suitable for machine learning models.

## Folder Structure

```
fraud-detection/
│
├── data/
│   ├── raw/                        # Raw datasets (e.g., Fraud_Data.csv, IpAddress_to_Country.csv)
│   ├── processed/                  # Processed datasets after cleaning and feature engineering
│
├── notebooks/                      # Jupyter notebooks for analysis and preprocessing
│   ├── 01_data_analysis.ipynb       # Data cleaning, EDA, feature engineering
│
├── scripts/                        # Python scripts for modular preprocessing tasks
│   ├── preprocess.py                # Script for handling missing values, removing duplicates
│   ├── feature_engineering.py       # Script for feature creation (e.g., transaction velocity, time-based features)
│
└── README.md                       # Project documentation
```

## Dependencies

Before running the code, ensure you have the following dependencies installed:

- Python 3.8+
- Pandas
- NumPy
- Matplotlib / Seaborn (for visualizations)
- Scikit-learn (for preprocessing tasks)

You can install the necessary libraries using the following command:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-task1.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fraud-detection-task1
   ```
3. Run the Jupyter notebooks or scripts to preprocess the data:
   ```bash
   jupyter notebook notebooks/01_data_analysis.ipynb
   ```
4. Alternatively, run the preprocessing scripts:
   ```bash
   python scripts/preprocess.py
   python scripts/feature_engineering.py
   ```

## Results

- After Task 1, the data will be cleaned and ready for model training.  
- Features like transaction frequency, velocity, and time-based variables will be available to enhance the fraud detection models.

## Next Steps

Task 2 will involve building and training various machine learning models to detect fraud using the processed data.

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

