LSTM-Based Stock Price Prediction
Microsoft (MSFT) Closing Price Forecasting using Deep Learning

🚀 Project Overview

This project focuses on predicting Microsoft (MSFT) stock closing prices using a Long Short-Term Memory (LSTM) neural network.
Stock markets are inherently time-dependent, making LSTM models ideal due to their ability to capture long-term temporal dependencies in sequential data.

The goal of this project is to:

Analyze historical stock data 📊

Perform exploratory data analysis (EDA)

Build a deep learning model using LSTM

Predict future closing prices

Visually compare actual vs predicted prices

🧠 Why LSTM?

Traditional machine learning models struggle with sequential dependencies.
LSTM networks, a special type of Recurrent Neural Network (RNN), excel at:

Learning from historical time series data

Avoiding vanishing gradient problems

Capturing long-term trends in stock prices

📂 Dataset Details

Company: Microsoft (MSFT)

Time Period: 2013 – 2018

Total Records: 1,259 trading days

Features:

Open

High

Low

Close

Volume

🔍 Exploratory Data Analysis (EDA)
✔️ Data Inspection

Checked for missing values and data types

Converted date column into datetime format

📈 Visualizations

Open vs Close Price Trend

Trading Volume Over Time

Correlation Heatmap of numeric features

These steps helped validate data quality and uncover relationships between features.

⚙️ Data Preprocessing

Selected Closing Price as the target variable

Applied StandardScaler for normalization

Used a 60-day sliding window to create time-series sequences

Split data into:

95% Training

5% Testing

🏗️ Model Architecture

The LSTM model was built using TensorFlow Keras Sequential API:

Input → LSTM (64 units, return_sequences=True)
      → LSTM (64 units)
      → Dense (128 units, ReLU)
      → Dropout (0.5)
      → Dense (1 output)

🔧 Model Configuration

Optimizer: Adam

Loss Function: Mean Absolute Error (MAE)

Metric: Root Mean Squared Error (RMSE)

Epochs: 20

Batch Size: 32

📊 Training Performance

Training loss consistently decreased across epochs

RMSE stabilized, indicating good learning behavior

No major overfitting observed

🔮 Predictions & Results

After training:

The model predicted closing prices on unseen test data

Predictions were inverse-transformed back to original price scale

Results were visualized alongside actual prices

📉 Final Output Visualization

Blue: Training (Actual)

Orange: Test (Actual)

Red: Model Predictions

The predicted curve closely follows the actual stock price trend, demonstrating strong time-series learning capability.

✅ Key Results

✔ Successfully captured market trends
✔ Accurate short-term price predictions
✔ Smooth alignment between actual and predicted values
✔ Robust LSTM architecture with minimal tuning

🛠️ Technologies Used

Python

TensorFlow / Keras

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

📌 Project Structure
📁 LSTM-Stock-Prediction
│── 📄 MicrosoftStock.csv
│── 📄 lstm_stock_prediction.ipynb
│── 📄 README.md

🚧 Future Improvements

Add multiple features (Open, High, Low, Volume)

Use GRU or Bidirectional LSTM

Perform hyperparameter tuning

Extend prediction horizon (multi-step forecasting)

Compare with ARIMA and Prophet models
