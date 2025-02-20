# Handwritten Digit Recognition Webapp

This project integrates a Flask-based web service with PyTorch to provide handwritten digit recognition. Users can draw digits directly on a web page for real-time analysis.

## Requirements

- Python >= 3.8
- Flask
- Flask-Cors
- PyTorch
- torchvision
- Pillow

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/tsaiJay/handwrite_digit_recognition_webapp.git
cd handwrite_digit_recognition_webapp
```

### 2. Set up a virtual environment

#### Using `venv`

You can use `venv` to create a virtual environment. Run the following command:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

- On macOS/Linux:

  ```bash
  source venv/bin/activate
  ```

Install the required packages:

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

#### Using Anaconda

If you prefer Anaconda, follow these steps:

```bash
conda create --name digit_recognition python=3.8
conda activate digit_recognition
pip install -r requirements.txt
```

### 3. Train and Evaluate the Model (Optional)

If you do not want to train the model yourself, skip this step. Otherwise, use the `train.py` script, which utilizes the MNIST dataset to train a simple CNN model.

```bash
cd ai_model
python train.py
```

The trained model will be saved as `lenet.pt` in the `ai_model/weight` directory. A training accuracy above 97% is considered sufficient.

After training, you can test the inference procedure of the model on a single MNIST dataset sample using the `eval.py` script:

```bash
python eval.py
```

### 4. Run the Server

Before starting the server, copy the trained model weights into the `web_server/model_weight` folder to ensure the backend predictor functions correctly.

Start the Flask server by running:

```bash
python server.py
```

The server will run at `http://127.0.0.1:5000`.

## API Endpoints

### **POST /predict**

This endpoint accepts a base64-encoded image of a handwritten digit and returns the predicted digit along with the confidence level.

<!-- ## Usage

You can use tools like Postman or `cURL` to test the `/predict` endpoint by sending a `POST` request with the base64-encoded image data.
 -->

Request json format
```json
{
    "image": "data:image/png;base64,<base64-image-data>"
}
```
<!-- 
Response json format
```json
{
    "confidence": 0.98  <<< may be add
    'success': True,
    'prediction': prediction
}
``` -->


## License

<!-- This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
 -->
You are free to share and adapt this work for educational and non-commercial purposes, provided that proper credit is given. For commercial use, please contact the project owner.


## ToDO
- [ ] add confident score
- [ ] black painting canvas