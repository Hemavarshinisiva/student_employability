import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    df = pd.read_excel("Student-Employability-Datasets.xlsx", sheet_name="Data")
except FileNotFoundError:
    raise FileNotFoundError("Dataset file not found. Please ensure the file is available in the same directory.")

X = df.iloc[:, 1:-2].values  # Adjust column selection if needed
y = (df["CLASS"] == "Employable").astype(int)  # Convert labels to binary (0 or 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron.fit(X_train_scaled, y_train)
y_pred = perceptron.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained successfully with accuracy: {accuracy:.2f}")

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("perceptron.pkl", "wb") as model_file:
    pickle.dump(perceptron, model_file)

def predict_employability(name, general_appearance, manner_of_speaking, physical_condition,
                           mental_alertness, self_confidence, ability_to_present_ideas,
                           communication_skills):
    try:
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        with open("perceptron.pkl", "rb") as model_file:
            perceptron = pickle.load(model_file)
    except FileNotFoundError:
        return "Error: Model or scaler not found. Please retrain the model."

    
    input_data = np.array([[general_appearance, manner_of_speaking, physical_condition,
                             mental_alertness, self_confidence, ability_to_present_ideas,
                             communication_skills]])
    input_scaled = scaler.transform(input_data)
    prediction = perceptron.predict(input_scaled)
    result = f"{name} is Employable - Congratulations! 🎉😊" if prediction[0] == 1 else f"{name} is Less Employable - Work Hard and Improve yourself! 💪"
    return result
inputs = [
    gr.Textbox(label="Name"),
    gr.Slider(1, 5, step=1, label="General Appearance"),
    gr.Slider(1, 5, step=1, label="Manner of Speaking"),
    gr.Slider(1, 5, step=1, label="Physical Condition"),
    gr.Slider(1, 5, step=1, label="Mental Alertness"),
    gr.Slider(1, 5, step=1, label="Self Confidence"),
    gr.Slider(1, 5, step=1, label="Ability to Present Ideas"),
    gr.Slider(1, 5, step=1, label="Communication Skills"),
]
output = gr.Textbox(label="Employability Prediction")
app = gr.Interface(fn=predict_employability, inputs=inputs, outputs=output, title="Employability Predictor", submit_btn="Get Yourself Evaluated🪄")
app.launch(share=True)
