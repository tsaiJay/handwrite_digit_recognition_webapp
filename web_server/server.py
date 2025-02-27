from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
from PIL import Image
import io

from predictor import Predictor

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

@app.route('/')
def home():
    return render_template('index.html')  # render index.html

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        # Get the image data from the request, and convert to PIL image
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove the "data:image/png;base64," part

        # Convert base64 to PIL image
        image_bytes = base64.b64decode(image_data)
        PIL_image = Image.open(io.BytesIO(image_bytes))
        
        # Image Preprocessing
        PIL_image = PIL_image.convert('L')
        PIL_image = PIL_image.resize((28, 28))
        # PIL_image.show()  # uncommand this line to show Processed Image locally

        # ML prediction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        prediction = digit_predictor.inference(PIL_image)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Initialize the digit classifier object
    digit_predictor = Predictor(weight_path='./model_weight/lenet.pt')  # model takes 1x28x28 input sizse

    # app.run(debug=True, port=5000) 
    app.run(debug=False, port=5000) 
