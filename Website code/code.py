from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)

# Function to load your models
def load_models():
    model_1 = tf.keras.models.load_model(r"C:\Users\bishn\Desktop\Website code\model_1.h5")
    model_2 = tf.keras.models.load_model(r"C:\Users\bishn\Desktop\Website code\model_2.h5")
    return model_1, model_2

# Load models when the script starts
model_1, model_2 = load_models()

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']  # Image file from form
        if image_file:
            image_location = os.path.join(
                r"C:\Users\bishn\Desktop\Website code\Input_images",
                secure_filename(image_file.filename)
            )
            image_file.save(image_location)

            # First Model part-------------------------------------------------------------------------------- 

            # Load image and resize to the correct dimensions
            image = tf.keras.preprocessing.image.load_img(image_location, target_size=(576, 928))
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Normalize the histogram of the grayscale image
            gray = cv2.equalizeHist(gray)
            
            # Convert back to color
            img_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            img_output = tf.keras.preprocessing.image.img_to_array(img_output)
            img_output = img_output / 255.0  # Normalize image

            # Predict segmentation mask
            img_output = tf.expand_dims(img_output, axis=0)  # Add batch dimension
            segmentation_mask = model_1.predict(img_output)

            segmentation_mask = segmentation_mask > 0.1
            segmentation_mask = segmentation_mask*255

            masked_image = img_output[0] * segmentation_mask[0] / 255.0
            # convert tensor to numpy array and make sure its values are in the range of 0-255
            masked_image_np = np.array(masked_image)
            masked_image_np = (masked_image_np * 255).astype(np.uint8)

            # convert numpy array to PIL Image
            prediction_1_image = Image.fromarray(masked_image_np)
            prediction_1_image.save(r'C:\Users\bishn\Desktop\Website code\Proccesing_image\prediction_1_output.png')

            # Second Model Part-------------------------------------------------------------------------------------------------

            img_path = r"C:\Users\bishn\Desktop\Website code\Proccesing_image\prediction_1_output.png"
            # Load image and resize to the correct dimensions
            image = Image.open(img_path)
            image = image.resize((64, 64))  # Resize image for first model
            image = tf.keras.preprocessing.image.img_to_array(image)
            img_tensor = np.expand_dims(image, axis=0)
            img_tensor /= 255.

            # Predictions from the second model
            prediction_2 = model_2.predict(img_tensor)
            # print("Printing prediction of model 2=",prediction_2)
            
            final_result = 'Normal' if prediction_2 < 0.5 else 'Pneumonia'

            return render_template('result.html', result=final_result, percentage=round(prediction_2[0][0] * 100, 2))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
