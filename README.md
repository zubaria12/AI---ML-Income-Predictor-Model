
# 💼 AI-Powered Income Class Predictor

This is a Python-based desktop application that predicts whether an individual's income exceeds $50K per year based on demographic and work-related attributes. It also provides **rule-based AI recommendations** to potentially increase income, and offers **visual insights** into the model's behavior using a sleek dark-themed GUI.

## 📌 Features

- 🧠 **Machine Learning Prediction**  
  Predicts income class using a trained Random Forest classifier on the UCI Adult Census dataset.

- 🤖 **AI-Powered Recommendations**  
  Personalized rule-based suggestions to help users boost their earning potential based on the model's interpretation.

- 📊 **Model Performance Insights**  
  Interactive visualizations including:
  - Feature importance
  - Prediction probability histograms
  - Confusion matrix
  - Correlation heatmaps
  - Cumulative probability curves

- 🖥️ **Interactive Desktop GUI**  
  User-friendly interface built with `tkinter`, styled with a custom Gemini-inspired dark theme.

## 🚀 How It Works

1. **Input Personal Information**  
   The user enters demographic details (e.g. age, education, occupation).

2. **Prediction Engine**  
   The Random Forest model predicts if the user earns `<=50K` or `>50K`.

3. **Recommendation Engine**  
   Based on the user input and feature contributions, the app suggests realistic, actionable strategies for income improvement.

4. **Visualization Dashboard**  
   The user can explore model performance and dataset insights with dynamic graphs and charts.

## 🛠️ Technologies Used

- **Python 3.9+**
- `scikit-learn` – Model training and prediction
- `pandas`, `numpy` – Data preprocessing
- `matplotlib`, `seaborn` – Visualizations
- `tkinter` – Desktop GUI
- `ttk` – Enhanced widget styling
- `matplotlib.backends.backend_tkagg` – Embedding plots into GUI


## 📂 Project Structure

.
├── main.py # Main application file with model, GUI, visualizations

├── README.md # You're here!



## 📦 Installation

1. Clone the repository:

git clone https://github.com/zubaria12/AI---ML-Income-Predictor-Model.git
cd income-predictor-ai

Install required packages:
scikit-learn
pandas
numpy
matplotlib
seaborn

Run the app:
python main.py
  
  
👩‍💻 Authors

Umar

Zubaria

📝 License
This project is for educational purposes. Please contact the authors for commercial use or further collaboration.

