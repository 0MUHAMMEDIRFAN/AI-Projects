import tensorflow as tf
import numpy as np

# Define required variables
img_size = (180, 180)
model = tf.keras.models.load_model('flower_model.h5')  # Replace with your model path
class_names = ['rose','another','cat']  # Replace with your actual class names

# Load and predict test image
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)
    print("Predictions:", predictions)
    print("Prediction:", class_names[np.argmax(score)], "| Confidence: {:.2f}%".format(100 * np.max(score)))

# Example usage
predict_image('flowersame1.jpeg')  # Replace with your test image path
