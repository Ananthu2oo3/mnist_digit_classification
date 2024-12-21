import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# model = tf.keras.models.load_model('mnist_model.pkl')
with open('mnist_model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_image(image_path):
    
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28)) 
    image_array = np.array(image)  
    image_array = image_array / 255.0  
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict_digit(image_path):
    
    processed_image = preprocess_image(image_path)
    
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    

    plt.imshow(processed_image[0], cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.axis('off')
    plt.show()
    
    return predicted_digit


image_path = '/home/ananthu/Downloads/8.jpeg'  
predicted_digit = predict_digit(image_path)
print(f"Model Prediction: {predicted_digit}")
