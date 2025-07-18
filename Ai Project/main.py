import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import datetime 
import csv 
from matplotlib.colors import to_rgb 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- PART 1: DATA PROCESSING ---

try:
    df = pd.read_csv("adult.csv")
except FileNotFoundError:
    print("Error: adult.csv not found. Please ensure the file is in the same directory.")
    exit(1)

# Handle missing values by replacing '?' with NaN and then dropping rows
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Drop 'fnlwgt' column if it exists, as it's often not relevant for prediction
if 'fnlwgt' in df.columns:
    df.drop(['fnlwgt'], axis=1, inplace=True)

# Define expected columns based on the UCI Adult dataset to ensure data integrity
expected_columns = ['age', 'workclass', 'education', 'education.num', 'marital.status',
                    'occupation', 'relationship', 'race', 'sex', 'capital.gain',
                    'capital.loss', 'hours.per.week', 'native.country', 'income']

# Verify all expected columns exist to prevent errors during encoding/scaling
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns in dataset: {missing_columns}")
    exit(1)

# Encode categorical variables into numerical format using LabelEncoder
label_encoders = {}
categorical_columns = df.select_dtypes(include='object').columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features (X) and target variable (y)
X = df.drop('income', axis=1)
y = df['income']

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load a pre-trained model or train a new one if not found
model_file = "income_model.pkl"
if not os.path.exists(model_file):
    print("No pre-trained model found. Training a new model...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    base_model = RandomForestClassifier(random_state=42)
    model = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1)
    model.fit(X_train, y_train)
    with open(model_file, 'wb') as f:
        pickle.dump(model.best_estimator_, f)
    print("Model training complete and saved.")
else:
    print("Loading pre-trained model...")
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Retraining model...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        base_model = RandomForestClassifier(random_state=42)
        model = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1)
        model.fit(X_train, y_train)
        with open(model_file, 'wb') as f:
            pickle.dump(model.best_estimator_, f)
        print("Model retrained and saved.")


# Evaluate the model's performance on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("Confusion Matrix (for debugging):\n", conf_matrix)
print("Feature Importances (for debugging):\n", model.feature_importances_)


# --- PART 2: GUI WITH TKINTER ---

app = tk.Tk()
app.title("Income Insights - AI Powered Prediction")
app.geometry("950x950") # Larger default size for a better initial experience

# --- Gemini Dark Theme Configuration ---
COLOR_GEMINI_BACKGROUND_PRIMARY = "#1A1A1A" # Deep charcoal
COLOR_GEMINI_BACKGROUND_SECONDARY = "#2C2C2C" # Slightly lighter for cards/sections
COLOR_GEMINI_BACKGROUND_TERTIARY = "#383838" # Even lighter for plot backgrounds
COLOR_GEMINI_TEXT_PRIMARY = "#E0E0E0" # Light gray for main text
COLOR_GEMINI_TEXT_SECONDARY = "#B0B0B0" # Muted gray for descriptions
COLOR_GEMINI_ACCENT_BLUE = "#5D9EEB" # A vibrant blue for primary actions/highlights
COLOR_GEMINI_ACCENT_BLUE_HOVER = "#4A8CD0" # Slightly darker blue for hover
COLOR_GEMINI_ACCENT_GREEN = "#76D7C4" # Teal/green for positive outcomes/success
COLOR_GEMINI_ACCENT_RED = "#FF6B6B" # Soft red for negative outcomes/errors
COLOR_GEMINI_ACCENT_ORANGE = "#FFC300" # Warm orange for warnings/highlights
COLOR_GEMINI_BORDER = "#444444" # Subtle border color

app.configure(bg=COLOR_GEMINI_BACKGROUND_PRIMARY)

style = ttk.Style()
style.theme_use('clam') # 'clam' is a good base for customization

# Define custom font family and sizes for a modern, clear look
FONT_FAMILY = ("Segoe UI", "Arial", "Helvetica")
plt.rcParams['font.family'] = FONT_FAMILY[0] # Ensure matplotlib uses this font family

FONT_APP_TITLE = (FONT_FAMILY, 24, "bold")
FONT_HEADER = (FONT_FAMILY, 18, "bold")
FONT_LABEL = (FONT_FAMILY, 13, "bold")
FONT_INPUT = (FONT_FAMILY, 12)
FONT_DESCRIPTION = (FONT_FAMILY, 10, "italic")
FONT_SMALL_BOLD = (FONT_FAMILY[0], 11, "bold") 

# Apply styles to ttk widgets
style.configure("TLabel", background=COLOR_GEMINI_BACKGROUND_PRIMARY, foreground=COLOR_GEMINI_TEXT_PRIMARY, font=FONT_LABEL)
style.configure("TButton", 
                background=COLOR_GEMINI_ACCENT_BLUE, 
                foreground=COLOR_GEMINI_TEXT_PRIMARY, # Text color for button (light)
                font=FONT_LABEL, 
                padding=15, 
                relief="flat", 
                borderwidth=0)
style.map("TButton", 
          background=[('active', COLOR_GEMINI_ACCENT_BLUE_HOVER), ('pressed', COLOR_GEMINI_ACCENT_BLUE_HOVER)], 
          foreground=[('active', COLOR_GEMINI_TEXT_PRIMARY), ('pressed', COLOR_GEMINI_TEXT_PRIMARY)])

style.configure("TEntry", 
                fieldbackground=COLOR_GEMINI_BACKGROUND_SECONDARY, # Input field background
                foreground=COLOR_GEMINI_TEXT_PRIMARY, 
                insertbackground=COLOR_GEMINI_TEXT_PRIMARY, 
                font=FONT_INPUT, 
                relief="solid", 
                borderwidth=1, 
                bordercolor=COLOR_GEMINI_BORDER)
style.configure("TCombobox", 
                fieldbackground=COLOR_GEMINI_BACKGROUND_SECONDARY, 
                background=COLOR_GEMINI_BACKGROUND_SECONDARY, # This applies to the dropdown button
                foreground=COLOR_GEMINI_TEXT_PRIMARY, 
                selectbackground=COLOR_GEMINI_ACCENT_BLUE, # Highlight color for selected item in dropdown
                selectforeground=COLOR_GEMINI_TEXT_PRIMARY, # Text color for selected item
                font=FONT_INPUT, 
                relief="solid", 
                borderwidth=1, 
                bordercolor=COLOR_GEMINI_BORDER)
style.map("TCombobox", 
          fieldbackground=[('readonly', COLOR_GEMINI_BACKGROUND_SECONDARY)], 
          selectbackground=[('readonly', COLOR_GEMINI_ACCENT_BLUE)]) 

# Custom style for Treeview
style.configure("Treeview", 
                background=COLOR_GEMINI_BACKGROUND_SECONDARY, 
                foreground=COLOR_GEMINI_TEXT_PRIMARY, 
                fieldbackground=COLOR_GEMINI_BACKGROUND_SECONDARY, 
                font=FONT_INPUT, 
                rowheight=32, 
                borderwidth=0)
style.configure("Treeview.Heading", 
                background=COLOR_GEMINI_ACCENT_BLUE, 
                foreground=COLOR_GEMINI_TEXT_PRIMARY, 
                font=FONT_SMALL_BOLD, 
                padding=(12, 10), 
                relief="flat")
style.map("Treeview", 
          background=[('selected', COLOR_GEMINI_ACCENT_BLUE)], 
          foreground=[('selected', COLOR_GEMINI_TEXT_PRIMARY)])
style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})]) # Essential for proper rendering

# --- Main Application Layout ---
app_title_label = tk.Label(app, text="AI Income Predictor", bg=COLOR_GEMINI_BACKGROUND_PRIMARY, fg=COLOR_GEMINI_ACCENT_BLUE, font=FONT_APP_TITLE)
app_title_label.pack(pady=(25, 15))

# Main frame to hold canvas and buttons
main_container_frame = tk.Frame(app, bg=COLOR_GEMINI_BACKGROUND_PRIMARY)
main_container_frame.pack(fill="both", expand=True, padx=25, pady=10)

# --- Scrollable Input Area ---
canvas_frame = tk.Frame(main_container_frame, bg=COLOR_GEMINI_BACKGROUND_PRIMARY)
canvas_frame.pack(fill="both", expand=True, side="top")

canvas = tk.Canvas(canvas_frame, bg=COLOR_GEMINI_BACKGROUND_PRIMARY, highlightthickness=0)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create a frame inside the canvas to hold all input widgets
input_widgets_frame = tk.Frame(canvas, bg=COLOR_GEMINI_BACKGROUND_PRIMARY)
canvas_window_id = canvas.create_window((0, 0), window=input_widgets_frame, anchor="nw")

# Function to adjust the width of the inner frame when the canvas resizes
def on_canvas_configure(event):
    canvas.itemconfig(canvas_window_id, width=event.width)
canvas.bind('<Configure>', on_canvas_configure)


# Input fields data structures
entries = {}
dropdowns = {}
numerical_fields = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_fields = ['workclass', 'education', 'marital.status', 'occupation',
                      'relationship', 'race', 'sex', 'native.country']
all_fields = numerical_fields + categorical_fields

# Descriptions for each input field (User Manual)
field_descriptions = {
    'age': "Enter your current age (e.g., 35). Valid range: 0-120 years.",
    'workclass': "Select your primary employment type (e.g., Private, Self-emp-not-inc).",
    'education': "Select your highest education level completed (e.g., Bachelors, HS-grad).",
    'education.num': "Numerical score of education (e.g., 9 for HS-grad, 13 for Bachelors). Valid range: 1-16.",
    'marital.status': "Select your current marital status (e.g., Married-civ-spouse, Never-married).",
    'occupation': "Select your job occupation (e.g., Exec-managerial, Tech-support).",
    'relationship': "Select your relationship status in a family (e.g., Husband, Not-in-family).",
    'race': "Select your racial group (e.g., White, Black).",
    'sex': "Select your biological sex (Male or Female).",
    'capital.gain': "Enter income from investments/assets (e.g., 0, 1000). Must be non-negative.",
    'capital.loss': "Enter losses from investments/assets (e.g., 0, 500). Must be non-negative.",
    'hours.per.week': "Enter hours worked per week (e.g., 40). Valid range: 0-168 hours.",
    'native.country': "Select your country of origin (e.g., United-States, Mexico)."
}

# Determine the maximum label length to ensure equal label column width
# Add some extra characters for padding for consistent length
max_label_len = max(len(field.replace('.', ' ').title()) for field in all_fields) + 5 

def create_input_field(label_text, row, field_name, field_type):
    """Creates a distinct frame for each input field, including label, description, and input widget."""
    # Reduced overall padding for a more compact card
    field_card_frame = tk.Frame(input_widgets_frame, bg=COLOR_GEMINI_BACKGROUND_SECONDARY, padx=8, pady=4, relief="solid", borderwidth=1, highlightbackground=COLOR_GEMINI_BORDER, highlightthickness=1)
    field_card_frame.grid(row=row, column=0, sticky="ew", pady=4, padx=10) # Reduced pady between cards
    input_widgets_frame.grid_columnconfigure(0, weight=1)

    # Use a nested frame for layout within the card
    # This frame will hold the label/description and the input widget
    content_frame = tk.Frame(field_card_frame, bg=COLOR_GEMINI_BACKGROUND_SECONDARY)
    content_frame.pack(fill="x", expand=True, pady=(5, 5)) # Padding for content inside the card

    # Configure columns within the content_frame
    content_frame.grid_columnconfigure(0, weight=1) # Column for label/description
    content_frame.grid_columnconfigure(1, weight=0) # Column for input widget (fixed width)

    # Main field label
    # Use grid on content_frame directly for better control
    tk.Label(content_frame, text=label_text, bg=COLOR_GEMINI_BACKGROUND_SECONDARY, font=FONT_LABEL, fg=COLOR_GEMINI_TEXT_PRIMARY, anchor="w", width=max_label_len).grid(row=0, column=0, pady=(0, 2), padx=5, sticky="w")

    # "What to Enter" description
    desc_text = field_descriptions.get(field_name, "No description available.")
    tk.Label(content_frame, text=f"What to Enter: {desc_text}", bg=COLOR_GEMINI_BACKGROUND_SECONDARY, font=FONT_DESCRIPTION, fg=COLOR_GEMINI_TEXT_SECONDARY, wraplength=400, justify="left", anchor="w").grid(row=1, column=0, pady=(0, 8), padx=5, sticky="ew") # Increased wraplength

    # **CRUCIAL ADJUSTMENT:** Significantly increased width for dropdowns
    # A common width for both to maintain alignment, but generous for dropdowns
    INPUT_WIDGET_WIDTH = 25 # Increased width for entries and comboboxes

    if field_type in numerical_fields:
        entry = ttk.Entry(content_frame, width=INPUT_WIDGET_WIDTH, style="TEntry") 
        entry.grid(row=0, column=1, rowspan=2, padx=(5, 5), pady=5, sticky="e") # Stick to 'e'ast
        return entry
    else:
        if field_name not in label_encoders:
            messagebox.showerror("Error", f"Field {field_name} not found in dataset for encoding.")
            app.destroy()
            exit(1)
        values = ["Select a value..."] + sorted(list(label_encoders[field_name].classes_))
        dropdown = ttk.Combobox(content_frame, values=values, width=INPUT_WIDGET_WIDTH, state="readonly", style="TCombobox") 
        dropdown.grid(row=0, column=1, rowspan=2, padx=(5, 5), pady=5, sticky="e") # Stick to 'e'ast
        dropdown.set("Select a value...")
        return dropdown

# Create input fields dynamically within the scrollable frame
current_row_idx = 0
for field in all_fields:
    if field in numerical_fields:
        entries[field] = create_input_field(field.replace('.', ' ').title(), current_row_idx, field, field)
    else:
        dropdowns[field] = create_input_field(field.replace('.', ' ').title(), current_row_idx, field, field)
    current_row_idx += 1

# Update scroll region after all widgets are added
input_widgets_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))


def validate_numerical_input(value, field):
    """Validates numerical input for specific ranges and types."""
    try:
        val = float(value)
        if field == 'age' and not (0 <= val <= 120):
            raise ValueError(f"{field.replace('.', ' ').title()} must be between 0 and 120.")
        if field == 'education.num' and not (1 <= val <= 16):
            raise ValueError(f"{field.replace('.', ' ').title()} must be between 1 and 16.")
        if field in ['capital.gain', 'capital.loss'] and val < 0:
            raise ValueError(f"{field.replace('.', ' ').title()} cannot be negative.")
        if field == 'hours.per.week' and not (0 <= val <= 168):
            raise ValueError(f"{field.replace('.', ' ').title()} must be between 0 and 168.")
        return val
    except ValueError as e:
        if str(e).startswith(field):
            raise e
        raise ValueError(f"{field.replace('.', ' ').title()} must be a valid number.")

def save_prediction_history(user_input_data, prediction_output):
    """
    Saves the user's input and the model's prediction to a CSV file.
    user_input_data: A dictionary of raw user inputs (e.g., {'age': 30, 'workclass': 'Private'}).
    prediction_output: A dictionary of prediction results (e.g., {'prediction': '>50K', 'estimated_income': 60000.0, 'confidence': 0.85}).
    """
    history_file = "prediction_history.csv"
    
    # Prepare header if file does not exist
    header_exists = os.path.exists(history_file)
    
    # Combine input data and output data
    record = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **user_input_data, # Unpack user input
        **prediction_output # Unpack prediction output
    }
    
    # Define the order of columns for the CSV
    # Ensure all input fields are included, then the prediction outputs
    column_order = ['timestamp'] + list(X.columns) + ['predicted_income_class', 'estimated_income', 'confidence']
    
    # Ensure all columns are present, fill missing with empty string if necessary
    row_data = {col: record.get(col, '') for col in column_order}

    try:
        with open(history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=column_order)
            if not header_exists:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Prediction saved to {history_file}")
    except IOError as e:
        print(f"Error saving prediction history: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving history: {e}")

# Global variables to store the last prediction's data for recommendations and assistant
last_user_input_raw = {}
last_prediction_class = ""
last_sorted_features = []
last_sorted_importances = []

def predict_income():
    """Gathers user input, predicts income, and displays results."""
    global last_user_input_raw, last_prediction_class, last_sorted_features, last_sorted_importances

    try:
        user_input_raw = {}
        for field in all_fields:
            if field in numerical_fields:
                val = entries[field].get().strip()
                if not val:
                    raise ValueError(f"'{field.replace('.', ' ').title()}' cannot be empty.")
                user_input_raw[field] = validate_numerical_input(val, field)
            else:
                val = dropdowns[field].get().strip()
                if not val or val == "Select a value...": # Check for placeholder
                    raise ValueError(f"Please select a value for '{field.replace('.', ' ').title()}'.")

                # Ensure case-insensitive matching for categorical values
                val_lower = val.lower()
                classes_lower = [c.lower() for c in label_encoders[field].classes_]
                if val_lower not in classes_lower:
                    raise ValueError(f"Invalid value '{val}' for '{field.replace('.', ' ').title()}'.")
                # Find the original case-sensitive class name
                for original in label_encoders[field].classes_:
                    if original.lower() == val_lower:
                        user_input_raw[field] = original
                        break
        
        # Convert raw input to numerical (for categorical) and then to a list for scaling
        processed_input_for_model = []
        for field in X.columns: # Iterate through model's expected feature order
            if field in numerical_fields:
                processed_input_for_model.append(user_input_raw[field])
            else:
                encoded_val = label_encoders[field].transform([user_input_raw[field]])[0]
                processed_input_for_model.append(encoded_val)

        user_input_scaled = scaler.transform([processed_input_for_model]) # Scale the input
        pred = model.predict(user_input_scaled)[0] # Get the prediction (0 or 1)
        pred_proba = model.predict_proba(user_input_scaled)[0][1] # Probability of >50K

        result_class = ">50K" if pred == 1 else "<=50K"

        # Heuristic income estimation based on probability and income class
        if pred == 1: # Income >50K
            income_estimate_min = 55000
            income_estimate_max = 300000 # Increased upper bound for >50K for more range
            income_estimate = income_estimate_min + (pred_proba * (income_estimate_max - income_estimate_min))
        else: # Income <=50K
            income_estimate_min = 18000 # Realistic lower bound for <=50K
            income_estimate_max = 50000
            income_estimate = income_estimate_min + (pred_proba * (income_estimate_max - income_estimate_min))
            if income_estimate > 50000: # Ensure estimate doesn't exceed 50K for <=50K predictions
                income_estimate = 50000


        # Feature contributions: Use model's feature importances
        feature_importances = model.feature_importances_
        # Sort features by importance for a more meaningful display
        sorted_idx = np.argsort(feature_importances)[::-1]
        sorted_features_current = [X.columns[i] for i in sorted_idx]
        sorted_importances_current = [feature_importances[i] for i in sorted_idx]

        # Store for global access
        last_user_input_raw = user_input_raw
        last_prediction_class = result_class
        last_sorted_features = sorted_features_current
        last_sorted_importances = sorted_importances_current

        # Prepare prediction output for saving
        prediction_output = {
            'predicted_income_class': result_class,
            'estimated_income': round(income_estimate, 2),
            'confidence': round(pred_proba, 4)
        }
        
        # Save search history
        save_prediction_history(user_input_raw, prediction_output)

        # Display results in a new window
        result_window = tk.Toplevel(app)
        result_window.title("Your Income Outlook - AI Prediction")
        result_window.geometry("800x700") # Generous size for results, excluding recommendations
        result_window.configure(bg=COLOR_GEMINI_BACKGROUND_SECONDARY) # Use secondary background for result window

        # Configure result window responsiveness
        result_window.grid_columnconfigure(0, weight=1)
        result_window.grid_rowconfigure(0, weight=0) # Title
        result_window.grid_rowconfigure(1, weight=0) # Main message
        result_window.grid_rowconfigure(2, weight=0) # Estimated Income
        result_window.grid_rowconfigure(3, weight=0) # Confidence
        result_window.grid_rowconfigure(4, weight=0) # Feature header
        result_window.grid_rowconfigure(5, weight=1) # Treeview (expands)
        result_window.grid_rowconfigure(6, weight=0) # General Advice

        # Main header
        tk.Label(result_window, text="Your Income Outlook", bg=COLOR_GEMINI_BACKGROUND_SECONDARY, font=FONT_HEADER, fg=COLOR_GEMINI_TEXT_PRIMARY).grid(row=0, column=0, pady=(20, 10), sticky="n")

        # Outcome Message - Dynamic and clear
        message_text = ""
        message_color = ""
        if result_class == ">50K":
            message_text = f"Great News! Based on your profile, our AI projects your annual income to be **above $50,000**."
            message_color = COLOR_GEMINI_ACCENT_GREEN # Green for positive outcome
        else:
            message_text = f"Our AI estimates your annual income to be **$50,000 or less**."
            message_color = COLOR_GEMINI_ACCENT_RED # Red for less favorable outcome

        tk.Label(result_window, text=message_text, bg=COLOR_GEMINI_BACKGROUND_SECONDARY, font=FONT_LABEL, fg=message_color, wraplength=700, justify="center").grid(row=1, column=0, pady=5, padx=20, sticky="ew")
        tk.Label(result_window, text=f"Estimated Annual Income: **${income_estimate:,.2f}**", bg=COLOR_GEMINI_BACKGROUND_SECONDARY, font=FONT_LABEL, fg=COLOR_GEMINI_ACCENT_ORANGE).grid(row=2, column=0, pady=5, padx=20, sticky="ew")
        tk.Label(result_window, text=f"Confidence in prediction (Probability of >50K): **{pred_proba:.2%}**", bg=COLOR_GEMINI_BACKGROUND_SECONDARY, font=FONT_INPUT, fg=COLOR_GEMINI_TEXT_SECONDARY).grid(row=3, column=0, pady=5, padx=20, sticky="ew")

        # Feature Impact Section - More prominent and informative
        tk.Label(result_window, text="Key Factors Influencing Your Prediction", bg=COLOR_GEMINI_BACKGROUND_SECONDARY, font=FONT_HEADER, fg=COLOR_GEMINI_TEXT_PRIMARY).grid(row=4, column=0, pady=(20, 10), sticky="ew")
        
        columns = ("Feature", "Influence Score")
        tree = ttk.Treeview(result_window, columns=columns, show="headings", height=8, style="Treeview")
        tree.heading("Feature", text="Feature")
        tree.heading("Influence Score", text="Influence Score (Higher is More Impactful)")
        tree.column("Feature", width=250, anchor="w")
        tree.column("Influence Score", width=250, anchor="e")
        tree.grid(row=5, column=0, pady=10, padx=20, sticky="nsew")

        # Insert top N feature importance data
        num_top_features = 8 # Display top 8 most important features for better insight
        for i in range(min(num_top_features, len(sorted_features_current))):
            tree.insert("", "end", values=(sorted_features_current[i].replace('.', ' ').title(), f"{sorted_importances_current[i]:.4f}"))

        # Add general advice based on prediction (this remains as it's general, not "recommendation")
        general_advice_text = ""
        if result_class == ">50K":
            general_advice_text = "Your profile aligns with higher income potential. To further enhance your financial trajectory, consider strategic investments or advanced skill development. Maintaining strong work ethic and continuous learning are key."
        else:
            general_advice_text = "To potentially increase your income, consider focusing on acquiring new skills, pursuing further education or certifications, exploring industries with higher earning potential, or optimizing your current work arrangements for increased productivity and value creation. Networking can also open new doors."

        tk.Label(result_window, text=general_advice_text, bg=COLOR_GEMINI_BACKGROUND_SECONDARY, fg=COLOR_GEMINI_TEXT_SECONDARY, font=FONT_DESCRIPTION, wraplength=700, justify="left").grid(row=6, column=0, pady=(15, 20), padx=20, sticky="ew")
        
    except Exception as e:
        messagebox.showerror("Input Error", f"Please check your input values:\n{str(e)}")


# --- REVERTED: Rule-based AI Recommendations ---
def generate_recommendations(user_input, income_class, sorted_features, sorted_importances):
    """
    Generates tailored recommendations based on user input, prediction,
    and feature importances using predefined rules.
    """
    recommendations = []
    
    # General advice based on prediction
    if income_class == "<=50K":
        recommendations.append("Overall Outlook: Your current profile suggests an income likely below $50K. Focus on strategic changes to boost your earning potential.")
    else: # >50K
        recommendations.append("Overall Outlook: Your profile aligns with higher income potential. Consider maintaining your current trajectory and exploring further growth opportunities.")
    
    recommendations.append("\nPersonalized Action Plan:")

    # Top features influencing the prediction are key
    top_features = [f for f, _ in zip(sorted_features, sorted_importances)]
    
    # Example rule for education.num
    edu_num = user_input.get('education.num', None)
    if edu_num is not None and edu_num < 13 and 'education.num' in top_features[:5]: # Below Bachelor's and important
        recommendations.append(
            f"- **Education/Skill Upgrade:** Your current education level (Education Num: {int(edu_num)}) is a significant factor. Consider pursuing a Bachelor's degree (Education Num 13) or advanced vocational training/certifications in high-demand fields. This often leads to higher-paying roles."
        )
    elif edu_num is not None and edu_num >= 13 and 'education.num' in top_features[:5] and income_class == "<=50K":
         recommendations.append(
            f"- **Leverage Existing Education:** With your education level (Education Num: {int(edu_num)}), explore roles that fully utilize your qualifications. Consider professional development or specialized certifications within your field to stand out."
        )

    # Example rule for hours.per.week
    hours_per_week = user_input.get('hours.per.week', None)
    if hours_per_week is not None and hours_per_week < 40 and 'hours.per.week' in top_features[:5]:
        recommendations.append(
            f"- **Work Hours Optimization:** Working fewer than 40 hours per week ({int(hours_per_week)} hrs) can limit income. If feasible, consider increasing your work hours or pursuing a full-time position to maximize earnings."
        )
    elif hours_per_week is not None and hours_per_week >= 40 and 'hours.per.week' in top_features[:5] and income_class == "<=50K":
        recommendations.append(
            f"- **Efficiency & Value:** You're working full-time ({int(hours_per_week)} hrs). Focus on increasing your productivity and value in your current role to justify higher compensation, or explore roles with better pay for similar hours."
        )

    # Example rule for occupation
    occupation = user_input.get('occupation', None)
    if occupation is not None and 'occupation' in top_features[:5]:
        # This part is harder without external data, but we can make general suggestions
        if occupation in ["Adm-clerical", "Handlers-cleaners", "Farming-fishing", "Other-service", "Priv-house-serv"] and income_class == "<=50K":
            recommendations.append(
                f"- **Career Transition/Upskilling:** Your '{occupation}' role might have a lower income ceiling. Research high-growth industries (e.g., tech, healthcare, finance) and identify skills needed for a career transition. Online courses or bootcamps can be a good starting point."
            )
        elif income_class == ">50K":
             recommendations.append(
                f"- **Career Advancement:** Your '{occupation}' is a strong asset. Seek leadership roles, specialize further, or explore opportunities within your industry that offer higher compensation."
            )

    # Example rule for capital.gain/loss
    capital_gain = user_input.get('capital.gain', 0)
    capital_loss = user_input.get('capital.loss', 0)
    if (capital_gain == 0 and capital_loss == 0) and ('capital.gain' in top_features[:5] or 'capital.loss' in top_features[:5]):
        recommendations.append(
            "- **Financial Investment:** If you have no capital gains or losses, consider learning about and prudently engaging in investments. Diversified investments can provide an additional stream of income over time."
        )
    elif (capital_gain > 0 or capital_loss > 0) and ('capital.gain' in top_features[:5] or 'capital.loss' in top_features[:5]):
        recommendations.append(
            f"- **Capital Management:** You have reported capital gains (${int(capital_gain):,}) or losses (${int(capital_loss):,}). Review your investment strategy. For gains, consider reinvestment. For losses, analyze causes and adjust for future financial health."
        )

    # General fallback if no specific rule applies well
    if len(recommendations) <= 2: # Only general outlook + header
        recommendations.append(
            "- **Continuous Learning & Networking:** Regardless of your current status, continuous skill development and professional networking are always beneficial for career growth and income potential."
        )
        recommendations.append(
            "- **Financial Planning:** Consult a financial advisor to create a personalized plan for savings, investments, and debt management to optimize your financial future."
        )

    return "\n".join(recommendations)

def show_recommendations_window():
    """Opens a new window to display AI-powered recommendations (rule-based)."""
    if not last_user_input_raw:
        messagebox.showinfo("Recommendations", "Please make a prediction first to generate personalized recommendations.")
        return

    reco_window = tk.Toplevel(app)
    reco_window.title("AI-Powered Recommendations for Income Boost")
    reco_window.geometry("750x600")
    reco_window.configure(bg=COLOR_GEMINI_BACKGROUND_SECONDARY)
    reco_window.grid_columnconfigure(0, weight=1)
    reco_window.grid_rowconfigure(0, weight=0) # Title
    reco_window.grid_rowconfigure(1, weight=1) # Text area for recommendations

    tk.Label(reco_window, text="AI-Powered Recommendations for Income Boost", bg=COLOR_GEMINI_BACKGROUND_SECONDARY, 
             font=FONT_HEADER, fg=COLOR_GEMINI_ACCENT_BLUE).grid(row=0, column=0, pady=(20, 10), sticky="n")

    recommendations_text_widget = scrolledtext.ScrolledText(reco_window, bg=COLOR_GEMINI_BACKGROUND_TERTIARY, fg=COLOR_GEMINI_TEXT_PRIMARY, 
                                            font=FONT_INPUT, wrap="word", relief="flat", highlightthickness=0, padx=15, pady=15)
    recommendations_text_widget.grid(row=1, column=0, pady=5, padx=20, sticky="nsew")

    # Generate recommendations using the rule-based function
    recommendations_content = generate_recommendations(
        last_user_input_raw, last_prediction_class, last_sorted_features, last_sorted_importances
    )
    recommendations_text_widget.insert(tk.END, recommendations_content)
    recommendations_text_widget.config(state=tk.DISABLED) # Make it read-only

# --- Removed AI Assistant functions and global variables ---
# assistant_window = None
# assistant_chat_display = None
# assistant_input_entry = None
# def send_message_to_assistant(): ... (removed)
# def _get_and_display_assistant_response(): ... (removed)
# def get_assistant_response_with_openai(): ... (removed)
# def show_ai_assistant(): ... (removed)


def show_visualizations():
    """Opens a new window to display model performance visualizations."""
    viz_window = tk.Toplevel(app)
    viz_window.title("Model Performance & Data Insights")
    viz_window.geometry("900x750") # Generous size
    viz_window.configure(bg=COLOR_GEMINI_BACKGROUND_SECONDARY)

    viz_window.grid_columnconfigure(0, weight=1)
    viz_window.grid_rowconfigure(0, weight=0) # Title
    viz_window.grid_rowconfigure(1, weight=1) # Insights Summary (expands)
    viz_window.grid_rowconfigure(2, weight=0) # Buttons (at bottom)

    tk.Label(viz_window, text="Comprehensive Model Insights", bg=COLOR_GEMINI_BACKGROUND_SECONDARY, font=FONT_HEADER, fg=COLOR_GEMINI_TEXT_PRIMARY).grid(row=0, column=0, pady=(20, 10), sticky="n")

    insights_summary = (
        f"**Model Accuracy:** {accuracy:.2%} - This indicates how often the model correctly predicts income class.\n\n"
        f"**Dataset Overview:**\n"
        f"  Total Samples: {len(df):,} individuals analyzed.\n"
        f"  Income Distribution: <=$50K: {len(df[df['income'] == 0]):,} samples ({len(df[df['income'] == 0]) / len(df):.1%})\n"
        f"  >$50K: {len(df[df['income'] == 1]):,} samples ({len(df[df['income'] == 1]) / len(df):.1%})"
    )
    # Using a text widget for better formatting of multi-line text with bold parts
    insights_text_widget = tk.Text(viz_window, bg=COLOR_GEMINI_BACKGROUND_SECONDARY, fg=COLOR_GEMINI_TEXT_PRIMARY, font=FONT_INPUT,
                                   height=8, wrap="word", relief="flat", highlightthickness=0)
    insights_text_widget.insert(tk.END, insights_summary)
    # Apply tags for bold and color. FONT_SMALL_BOLD is a tuple, use it correctly.
    insights_text_widget.tag_configure("bold", font=(FONT_SMALL_BOLD[0], FONT_SMALL_BOLD[1], FONT_SMALL_BOLD[2]), foreground=COLOR_GEMINI_ACCENT_BLUE) 
    insights_text_widget.tag_add("bold", "1.0", "1.18") # Model Accuracy
    insights_text_widget.tag_add("bold", "3.0", "3.19") # Dataset Overview
    insights_text_widget.config(state=tk.DISABLED) # Make it read-only
    insights_text_widget.grid(row=1, column=0, pady=10, padx=25, sticky="nsew")


    # Navigation buttons for individual graphs
    button_frame = tk.Frame(viz_window, bg=COLOR_GEMINI_BACKGROUND_SECONDARY)
    button_frame.grid(row=2, column=0, pady=25, sticky="s") # Stick to bottom

    # Ensure buttons are responsive in the frame
    for i in range(5): # 5 buttons
        button_frame.grid_columnconfigure(i, weight=1)

    ttk.Button(button_frame, text="Feature Importance", command=show_feature_importance, style="TButton").grid(row=0, column=0, padx=10, pady=8)
    ttk.Button(button_frame, text="Probability Histogram", command=show_probability_histogram, style="TButton").grid(row=0, column=1, padx=10, pady=8)
    ttk.Button(button_frame, text="Confusion Matrix", command=show_confusion_matrix, style="TButton").grid(row=0, column=2, padx=10, pady=8)
    ttk.Button(button_frame, text="Correlation Heatmap", command=show_correlation_heatmap, style="TButton").grid(row=0, column=3, padx=10, pady=8)
    ttk.Button(button_frame, text="Cumulative Probability", command=show_cumulative_probability, style="TButton").grid(row=0, column=4, padx=10, pady=8)


def create_graph_window(title, geometry):
    """Helper function to create a new Toplevel window for graphs with new theme."""
    graph_window = tk.Toplevel(app)
    graph_window.title(title)
    graph_window.geometry(geometry)
    graph_window.configure(bg=COLOR_GEMINI_BACKGROUND_SECONDARY) # Use secondary background for the window
    graph_window.grid_columnconfigure(0, weight=1)
    graph_window.grid_rowconfigure(0, weight=1) # Make canvas expand
    return graph_window

# --- Matplotlib Plotting Functions (with Gemini Dark Theme and Enhanced Aesthetics) ---
def setup_plot_style(fig, ax, title):
    """Applies common Gemini dark theme styles to a matplotlib plot."""
    fig.patch.set_facecolor(COLOR_GEMINI_BACKGROUND_TERTIARY) # Darker plot background
    ax.set_facecolor(COLOR_GEMINI_BACKGROUND_TERTIARY)

    ax.set_title(title, fontsize=15, fontweight="bold", color=COLOR_GEMINI_TEXT_PRIMARY) # Bright title
    ax.tick_params(axis='x', colors=COLOR_GEMINI_TEXT_SECONDARY, labelsize=10) # Muted ticks
    ax.tick_params(axis='y', colors=COLOR_GEMINI_TEXT_SECONDARY, labelsize=10)

    # Set spine colors to a subtle light gray for visibility
    ax.spines['bottom'].set_color(COLOR_GEMINI_BORDER)
    ax.spines['top'].set_color(COLOR_GEMINI_BORDER)
    ax.spines['right'].set_color(COLOR_GEMINI_BORDER)
    ax.spines['left'].set_color(COLOR_GEMINI_BORDER)
    
    # Set label colors to primary light text
    ax.xaxis.label.set_color(COLOR_GEMINI_TEXT_PRIMARY)
    ax.yaxis.label.set_color(COLOR_GEMINI_TEXT_PRIMARY)

# Helper function to determine text color based on background luminance
def get_text_color(background_rgb):
    """
    Determines if white or black text is better for a given RGB background color.
    Uses the W3C luminance formula for perceived brightness.
    """
    r, g, b = background_rgb
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    # Use a threshold to decide between light and dark text
    return "white" if luminance < 0.5 else "black"


def show_feature_importance():
    graph_window = create_graph_window("Feature Importance", "800x600")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    setup_plot_style(fig, ax, "Feature Importance: What Drives Income?")

    feature_importance = model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    ax.bar(features[sorted_idx], feature_importance[sorted_idx], color=COLOR_GEMINI_ACCENT_GREEN, edgecolor=COLOR_GEMINI_BACKGROUND_TERTIARY, linewidth=1.5)
    
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Importance Score", fontsize=12)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([f.replace('.', ' ').title() for f in features[sorted_idx]],
                       rotation=45, ha="right",
                       fontfamily=FONT_SMALL_BOLD[0], 
                       fontsize=FONT_SMALL_BOLD[1])
    
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew", pady=15, padx=15)
    canvas_widget.config(bd=2, relief="solid", highlightbackground=COLOR_GEMINI_ACCENT_BLUE, highlightthickness=2)
    canvas.draw()
    plt.close(fig)

def show_probability_histogram():
    graph_window = create_graph_window("Prediction Probability Histogram", "800x600")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    setup_plot_style(fig, ax, "Distribution of Predicted Probabilities (>50K)")

    proba = model.predict_proba(X_test)[:, 1]
    ax.hist(proba, bins=20, color=COLOR_GEMINI_ACCENT_ORANGE, edgecolor=COLOR_GEMINI_BACKGROUND_TERTIARY, alpha=0.9, linewidth=1.5)
    ax.set_xlabel("Probability of Earning >50K", fontsize=12)
    ax.set_ylabel("Number of Individuals", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color=COLOR_GEMINI_BORDER) 
    
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew", pady=15, padx=15)
    canvas_widget.config(bd=2, relief="solid", highlightbackground=COLOR_GEMINI_ACCENT_BLUE, highlightthickness=2)
    canvas.draw()
    plt.close(fig)

def show_confusion_matrix():
    graph_window = create_graph_window("Confusion Matrix", "750x650")

    fig, ax = plt.subplots(figsize=(7, 6))
    setup_plot_style(fig, ax, "Model Performance: Confusion Matrix")

    # Draw the heatmap without specific annotation color initially
    # sns.heatmap returns the axes object
    heatmap_plot = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', 
                                xticklabels=['Predicted <=50K', 'Predicted >50K'], yticklabels=['Actual <=50K', 'Actual >50K'], ax=ax,
                                linewidths=.7, linecolor=COLOR_GEMINI_BACKGROUND_SECONDARY, 
                                cbar_kws={'label': 'Number of Samples'}) 
    
    # Dynamically set annotation color
    # Retrieve the color map and its normalization object
    cmap = plt.colormaps['YlGnBu'] # Use the modern way to get colormaps
    # The normalization for Confusion Matrix is typically based on the min/max of its values
    norm = plt.Normalize(vmin=conf_matrix.min(), vmax=conf_matrix.max())

    # Iterate over the text objects (annotations) in the heatmap
    for text_object in heatmap_plot.texts:
        # Get the numerical value associated with this text object
        # This value is what the colormap uses to determine the cell's color
        cell_value = int(text_object.get_text()) 
        
        # Get the color from the colormap based on the value
        cell_rgb = cmap(norm(cell_value))[:3] # [:3] to get R, G, B only

        # Set the text color based on cell luminosity
        text_object.set_color(get_text_color(cell_rgb))
        text_object.set_fontsize(14) # Retain font size

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(COLOR_GEMINI_TEXT_PRIMARY)
    cbar.ax.tick_params(colors=COLOR_GEMINI_TEXT_SECONDARY)
    
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew", pady=15, padx=15)
    canvas_widget.config(bd=2, relief="solid", highlightbackground=COLOR_GEMINI_ACCENT_BLUE, highlightthickness=2)
    canvas.draw()
    plt.close(fig)

def show_correlation_heatmap():
    graph_window = create_graph_window("Correlation Heatmap", "850x700")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    setup_plot_style(fig, ax, "Correlation Between Numerical Features")

    num_cols = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    corr_matrix = df[num_cols].corr()
    
    # Draw the heatmap without specific annotation color initially
    heatmap_plot = sns.heatmap(corr_matrix, annot=True, cmap='viridis', 
                                ax=ax, fmt=".2f", linewidths=.7, linecolor=COLOR_GEMINI_BACKGROUND_SECONDARY) 
    
    # Dynamically set annotation color
    # Get the colormap and its normalization object
    cmap = plt.colormaps['viridis'] # Modern way to get colormap
    norm = plt.Normalize(vmin=corr_matrix.min().min(), vmax=corr_matrix.max().max())

    # Iterate over the text objects (annotations) in the heatmap
    # heatmap_plot.texts is a flat list of all annotation text objects
    num_rows, num_cols_in_corr = corr_matrix.shape
    for i, text_object in enumerate(heatmap_plot.texts):
        # Calculate row and column indices for the current text_object
        # The texts are added in row-major order (row 0, then row 1, etc.)
        # There are N columns, so text_object at index 'i' corresponds to row i // N and col i % N
        row_idx = i // num_cols_in_corr
        col_idx = i % num_cols_in_corr
        
        # Get the numerical value from the correlation matrix for this cell
        cell_value = corr_matrix.iloc[row_idx, col_idx]
            
        # Get the color from the colormap based on the value
        cell_rgb = cmap(norm(cell_value))[:3] # [:3] to get R, G, B only

        # Set the text color based on cell luminosity
        text_object.set_color(get_text_color(cell_rgb))

    ax.tick_params(axis='x', colors=COLOR_GEMINI_TEXT_SECONDARY)
    ax.tick_params(axis='y', colors=COLOR_GEMINI_TEXT_SECONDARY)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(COLOR_GEMINI_TEXT_PRIMARY)
    cbar.ax.tick_params(colors=COLOR_GEMINI_TEXT_SECONDARY)

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew", pady=15, padx=15)
    canvas_widget.config(bd=2, relief="solid", highlightbackground=COLOR_GEMINI_ACCENT_BLUE, highlightthickness=2)
    canvas.draw()
    plt.close(fig)

def show_cumulative_probability():
    graph_window = create_graph_window("Cumulative Probability Plot", "800x600")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    setup_plot_style(fig, ax, "Cumulative Probability of Earning >50K")

    proba_sorted = np.sort(model.predict_proba(X_test)[:, 1])
    cumulative_proba = np.cumsum(proba_sorted) / np.sum(proba_sorted)
    ax.plot(range(len(proba_sorted)), cumulative_proba, color=COLOR_GEMINI_ACCENT_BLUE, linewidth=2.5)
    
    ax.set_xlabel("Individuals Ranked by Probability (from Low to High)", fontsize=12)
    ax.set_ylabel("Cumulative Probability Share", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5, color=COLOR_GEMINI_BORDER) 
    
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky="nsew", pady=15, padx=15)
    canvas_widget.config(bd=2, relief="solid", highlightbackground=COLOR_GEMINI_ACCENT_BLUE, highlightthickness=2)
    canvas.draw()
    plt.close(fig)


# --- Main Application Buttons and Footer ---
button_frame_main = tk.Frame(app, bg=COLOR_GEMINI_BACKGROUND_PRIMARY)
button_frame_main.pack(fill="x", padx=25, pady=(10, 5), side="bottom")
button_frame_main.grid_columnconfigure(0, weight=1)
button_frame_main.grid_columnconfigure(1, weight=1)
ttk.Button(button_frame_main, text="Predict My Income", command=predict_income, style="TButton").grid(row=0, column=0, padx=10, pady=10)
ttk.Button(button_frame_main, text="View Model Performance", command=show_visualizations, style="TButton").grid(row=0, column=1, padx=10, pady=10)
ttk.Button(button_frame_main, text="AI Recommendations", command=show_recommendations_window, style="TButton").grid(row=1, column=0, columnspan=2, padx=10, pady=10) # Centered
tk.Label(app, text="AI Project by Umar & Zubaria", bg=COLOR_GEMINI_BACKGROUND_PRIMARY, fg=COLOR_GEMINI_TEXT_SECONDARY, font=FONT_DESCRIPTION).pack(side="bottom", pady=(5, 15))
app.mainloop()