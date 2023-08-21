import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load trained model
model = keras.models.load_model("brainTumor_classification.h5")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def classify_image(image):
    image = tf.image.resize(image, (img_height, img_width))
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    scores = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(scores)]
    confidence = 100 * np.max(scores)
    return f"This image most likely belongs to {predicted_class} with a {confidence:.2f} percent confidence."

# Create the Gradio interface
input_image = gr.inputs.Image(shape=(img_height, img_width))
output_text = gr.outputs.Textbox()

gr.Interface(
    fn=classify_image,
    inputs=input_image,
    outputs=output_text,
    live=True,
    title="Brain Tumor Classification",
    description="Upload an image to classify the brain tumor type."
).launch()
