
# 🚗 Car Price Predictor

This app predicts the **car purchase price** of a customer based on financial and demographic input features using a trained **Artificial Neural Network (ANN)** model.

🔍 **How it works:**
- The model is trained using a dataset of customer purchase data.
- Inputs like age, annual salary, credit card debt, and net worth are used to predict the likely price a customer would pay for a car.
- The model scales input features, makes a prediction using a Keras model, and rescales the output to the original price format.

🧠 **Model Details:**
- Keras Sequential ANN
- Layers: Dense + ReLU with Dropout
- Loss: Mean Squared Error
- Optimizer: Adam
- Trained on scaled data

📦 **Files in this repo:**
- `app.py` – Gradio UI and prediction logic
- `car_price_ann_model.h5` – Pre-trained ANN model (upload separately)
- `car_purchasing.csv` – Dataset to refit scalers (upload separately)
- `requirements.txt` – List of required Python packages

🛠️ **How to use:**
1. Clone or upload these files to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Make sure to upload:
   - `app.py`
   - `car_price_ann_model.h5`
   - `car_purchasing.csv`
   - `requirements.txt`
3. Wait for the build. The app will launch with an interactive interface.

---

Made with ❤️ using [Gradio](https://gradio.app) and [TensorFlow](https://www.tensorflow.org/).
