from flask import Flask, render_template, request, jsonify, send_file
import pickle
from werkzeug.utils import secure_filename
import os
from io import BytesIO

model = None
filename = "models/model_nst.pkl"

app = Flask(__name__)

# Définir le dossier pour enregistrer les images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def render_default():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    # Récupérer les données du formulaire
    form_data = request.form
    
    # Récupérer le numéro
    number = form_data.get("number", None)
    if number is None:
        return jsonify({"error": "Le numéro est manquant"}), 400
    
    # Récupérer les images
    images = request.files.getlist("images")
    if len(images) < 2:
        return jsonify({"error": "Vous devez envoyer au moins deux images"}), 400
    
    # Appliquer le style de transfert
    stylised_image = model(images[0], images[1], iterations=int(number))
    
    # Envoyer l'image stylisée
    buf = BytesIO()
    stylised_image.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    model = pickle.load(open(filename, 'rb'))
    app.run(host="127.0.0.1", port=8000, threaded=True, debug=True)