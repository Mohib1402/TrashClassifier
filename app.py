import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Load your frozen model
model = tf.keras.models.load_model("final_trashnet_transfer_learning_model.keras")

# Mapping of original classes to broader categories
class_mapping = {
    0: "Compostable",  # compostable
    1: "Recyclables",  # recyclable
    2: "Trash",  #trash
}
# Define a function to preprocess the input image
def preprocess_image(image):
    # Resize the image to 128*128 (as required by your model)
    image = image.resize((128, 128))
    # Convert the image to a NumPy array and normalize it
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    # Ensure the image has the correct shape (32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array


# Define the prediction function
def classify_trash(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    print(predictions)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class = class_mapping[class_index]
    return f"Predicted Category: {predicted_class}", f"Confidence: {confidence*100:.2f}"

# Define the Gradio interface
interface = gr.Interface(
    fn=classify_trash,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[gr.Textbox(label="Predicted Category"), gr.Textbox(label="Confidence")],
    title="Trash Classifier",
    description="Upload an image of trash, and the model will classify it into 'Compostable', 'Recyclables' and 'Trash' based on its category."
)

# Run the app
if __name__ == "__main__":
    interface.launch()
