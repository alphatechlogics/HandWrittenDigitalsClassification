# 🧮 MNIST Digit Recognition Project

Welcome to the **MNIST Digit Recognition** project! This repository demonstrates a complete workflow for building, training, and deploying a neural network to recognize handwritten digits from the MNIST dataset. Additionally, it includes an interactive **Streamlit web app** for showcasing the model's capabilities.
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
---

## ✨ Features

- 🧠 **Model Training**: Train a neural network to classify handwritten digits with high accuracy.
- 🎨 **Interactive Drawing**: Draw digits directly on a canvas for real-time classification.
- 📂 **Image Upload**: Upload digit images for prediction.
- 💾 **Model Reusability**: Save the trained model and reload it for predictions.
- 📊 **Model Performance Metrics**: Display detailed training and evaluation results.

---

## 🛠️ Technologies Used

- **Python 3.10+**: Programming language.
- **TensorFlow/Keras**: Deep learning framework for model building and training.
- **Streamlit**: Web app framework for creating the interactive user interface.
- **NumPy**: Numerical computing library.
- **Matplotlib**: Data visualization for debugging and display.
- **OpenCV**: Image processing for resizing and normalization.

---

## 🚀 Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/alphatechlogics/HandWrittenDigitalsClassification.git
   cd mnist-digit-recognition
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

## 🧑‍💻 Model Training Workflow

- **Data Preprocessing:**

  - Normalize pixel values to [0, 1] for faster convergence.
  - Reshape input to match the neural network's requirements.

- **Model Architecture:**

  - A sequential neural network with:
    - Input layer to flatten the 28x28 grayscale image.
    - Fully connected dense layers with ReLU activation and dropout for regularization.
    - Output layer with softmax activation for class probabilities.

- **Training:**

  - Optimizer: Adam with a learning rate of 0.0003.
  - Loss Function: Sparse Categorical Crossentropy.
  - Metrics: Accuracy.

- **Model Saving:**

  - The trained model is saved in HDF5 format (`model.h5`).

## 🌟 Streamlit App Features

- **Sidebar Features:**

  - 🎥 Lottie Animation for visual engagement.
  - 📋 Information about the model and its features.
  - 📧 Contact section for support or inquiries.

- **Interactive Prediction:**

  - Choose between **drawing a digit** on a canvas or **uploading an image**.
  - Real-time predictions with confidence levels.
  - Easy-to-use buttons for classification.

## 📊 Results

- **Training Accuracy**: Achieved ~99% accuracy on the training dataset.
- **Validation Accuracy**: ~98% on the validation dataset.
- **Test Accuracy**: ~98.3% on the unseen test data.
