from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
# from ai_model.digit_classifier import DigitClassifierService

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Initialize the digit classifier service
digit_classifier = DigitClassifierService()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove the "data:image/png;base64," part
        
        # Convert base64 to image (you'll need this for ML processing later)
        # image_bytes = base64.b64decode(image_data)
        
        # TODO: Add your ML prediction logic here
        # For now, we'll just return a dummy response
        prediction = "Test Response: Image Received!"
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/classify', methods=['POST'])
def classify_digit():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        result = digit_classifier.predict(request.json['image'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 