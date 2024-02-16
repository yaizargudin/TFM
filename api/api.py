from flask import Flask, jsonify,request
import spacy
import joblib
import re

app = Flask(__name__)
nlp = spacy.load("es_core_news_sm")

def normalize_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres de puntuación
    text = re.sub(r'[^\w\s]', '', text)
    # Procesar el texto con spaCy
    doc = nlp(text)
    # Obtener lemas de cada token en el texto
    lemmas = [token.lemma_ for token in doc]
    # Unir tokens nuevamente en texto
    text = ' '.join(lemmas)
    return text

def set_result(prediction):
    if prediction == 0:
        return "Es correcto"
    elif prediction == 1:
        return "Falta el rol"
    elif prediction == 2:
        return "Falta el objetivo"
    elif prediction == 3:
        return "Falta la necesidad"
    elif prediction == 4:
        return "No es una historia de usuario"

@app.route('/')
def get_result():
    story = request.args.get('story')
    story = normalize_text(story)
    loaded_model = joblib.load('modelo_svm.joblib')
    predictions = loaded_model.predict([story])
    message = set_result(predictions[0])
    response = jsonify({'message': message})
    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

if __name__ == '__main__':
    app.run(debug=True)
