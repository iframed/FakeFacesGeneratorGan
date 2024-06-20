from flask import Flask, request, jsonify, send_file, Response, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import io
from PIL import Image
import base64

app = Flask(__name__)

# Load your models
discriminator_model = load_model('discriminator_model.h5')
generator_model = load_model('generator_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate():
    num_images = 1  
    input_noise = np.random.normal(0, 1, (num_images, 100)) 
    generated_images = generator_model.predict(input_noise)
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)  

    # Convert the generated image to PNG and return as a response
    img = generated_images[0]
    img_pil = Image.fromarray(img)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    buffered.seek(0)

    return send_file(buffered, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
