
import gradio as gr
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load model
model = load_model("car_price_ann_model.h5")

# Load dataset for scaler fitting
df = pd.read_csv("car_purchasing.csv", encoding='latin-1')
df.drop(['customer name', 'customer e-mail', 'country'], axis=1, inplace=True)
X = df.drop("car purchase amount", axis=1)
y = df["car purchase amount"]

# Fit scalers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# Input features
feature_names = list(X.columns)

# Create Gradio inputs
def create_inputs():
    components = []
    for f in feature_names:
        components.append(gr.Number(label=f))
    return components

# Prediction function
def predict_price(*inputs):
    inputs_array = np.array(inputs, dtype=np.float64).reshape(1, -1)
    inputs_scaled = scaler.transform(inputs_array)
    pred_scaled = model.predict(inputs_scaled)[0][0]
    pred_price = y_scaler.inverse_transform([[pred_scaled]])[0][0]
    return f"Predicted Car Purchase Price: ₹{pred_price:,.2f}"

# Launch Gradio app
input_components = create_inputs()
demo = gr.Interface(fn=predict_price,
                    inputs=input_components,
                    outputs="text",
                    title="Car Price Predictor",
                    description="Enter customer details to predict car purchase price.")
demo.launch()
