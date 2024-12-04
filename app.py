import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your frozen model
model = tf.keras.models.load_model("final_model_sgd_extended-2.keras")

# Mapping of original classes to broader categories
class_mapping = {
    0: "cardboard",  # cardboard
    1: "glass",  # glass
    2: "metal",  # metal
    3: "paper",  # paper
    4: "plastic",  # plastic
    5: "garbage"   # trash
}

# Define a function to preprocess the input image
def preprocess_image(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:  # Grayscale image
        image_array = np.stack((image_array,) * 3, axis=-1)
    return np.expand_dims(image_array, axis=0)

# Define the prediction function
def classify_trash(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    print(predictions)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class = class_mapping[class_index]
    return f"Predicted Category: {predicted_class}", f"Confidence: {confidence:.2f}"

# Define the Gradio interface
interface = gr.Interface(
    fn=classify_trash,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[gr.Textbox(label="Predicted Category"), gr.Textbox(label="Confidence")],
    title="Trash Classifier",
    description="Upload an image of trash, and the model will classify it into 'recycle' or 'garbage' based on its category."
)

# Run the app
if __name__ == "__main__":
    interface.launch()
