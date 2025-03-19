import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
MODEL_PATH = 'NANSNET_model.h5'  # Ensure the model file is in the same directory
model = load_model(MODEL_PATH)

# Dog breed class labels
class_labels = {
    0: "Chihuahua", 1: "Japanese_spaniel", 2: "Maltese_dog", 3: "Pekinese", 4: "Shih-Tzu",
    5: "Blenheim_spaniel", 6: "Papillon", 7: "Toy_terrier", 8: "Rhodesian_ridgeback", 9: "Afghan_hound",
    10: "Basset", 11: "Beagle", 12: "Bloodhound", 13: "Bluetick", 14: "Black-and-tan_coonhound",
    15: "Walker_hound", 16: "English_foxhound", 17: "Redbone", 18: "Borzoi", 19: "Irish_wolfhound",
    20: "Italian_greyhound", 21: "Whippet", 22: "Ibizan_hound", 23: "Norwegian_elkhound", 24: "Otterhound",
    25: "Saluki", 26: "Scottish_deerhound", 27: "Weimaraner", 28: "Staffordshire_bullterrier", 29: "American_Staffordshire_terrier",
    30: "Bedlington_terrier", 31: "Border_terrier", 32: "Kerry_blue_terrier", 33: "Irish_terrier", 34: "Norfolk_terrier",
    35: "Norwich_terrier", 36: "Yorkshire_terrier", 37: "Wire-haired_fox_terrier", 38: "Lakeland_terrier", 39: "Sealyham_terrier",
    40: "Airedale", 41: "Cairn", 42: "Australian_terrier", 43: "Dandie_Dinmont", 44: "Boston_bull",
    45: "Miniature_schnauzer", 46: "Giant_schnauzer", 47: "Standard_schnauzer", 48: "Scotch_terrier", 49: "Tibetan_terrier",
    50: "Silky_terrier", 51: "Soft-coated_wheaten_terrier", 52: "West_Highland_white_terrier", 53: "Lhasa", 54: "Flat-coated_retriever",
    55: "Curly-coated_retriever", 56: "Golden_retriever", 57: "Labrador_retriever", 58: "Chesapeake_Bay_retriever", 59: "German_short-haired_pointer",
    60: "Vizsla", 61: "English_setter", 62: "Irish_setter", 63: "Gordon_setter", 64: "Brittany_spaniel",
    65: "Clumber", 66: "English_springer", 67: "Welsh_springer_spaniel", 68: "Cocker_spaniel", 69: "Sussex_spaniel",
    70: "Irish_water_spaniel", 71: "Kuvasz", 72: "Schipperke", 73: "Groenendael", 74: "Malinois",
    75: "Briard", 76: "Kelpie", 77: "Komondor", 78: "Old_English_sheepdog", 79: "Shetland_sheepdog",
    80: "Collie", 81: "Border_collie", 82: "Bouvier_des_Flandres", 83: "Rottweiler", 84: "German_shepherd",
    85: "Doberman", 86: "Miniature_pinscher", 87: "Greater_Swiss_Mountain_dog", 88: "Bernese_mountain_dog", 89: "Appenzeller",
    90: "EntleBucher", 91: "Boxer", 92: "Bull_mastiff", 93: "Tibetan_mastiff", 94: "French_bulldog",
    95: "Great_Dane", 96: "Saint_Bernard", 97: "Eskimo_dog", 98: "Malamute", 99: "Siberian_husky",
    100: "Affenpinscher", 101: "Basenji", 102: "Pug", 103: "Leonberg", 104: "Newfoundland",
    105: "Great_Pyrenees", 106: "Samoyed", 107: "Pomeranian", 108: "Chow", 109: "Keeshond",
    110: "Brabancon_griffon", 111: "Pembroke", 112: "Cardigan", 113: "Toy_poodle", 114: "Miniature_poodle",
    115: "Standard_poodle", 116: "Mexican_hairless", 117: "Dingo", 118: "Dhole", 119: "African_hunting_dog"
}

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

# Function to predict dog breed
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    predicted_label = class_labels.get(predicted_class, "Unknown")
    return predicted_label

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded image
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Predict the image class
        predicted_label = predict_image(img_path)

        return jsonify({'result': predicted_label})

    return jsonify({'error': 'Failed to process the image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
