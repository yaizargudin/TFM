from sklearn import svm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy
import joblib

nlp = spacy.load("es_core_news_sm")

# Función de normalización de texto
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

# Cargar el conjunto de datos
df = pd.read_excel('datos.xlsx', index_col=None)
df.columns = ['Historia de usuario', 'Bien formada']
df['Historia de usuario'] = df['Historia de usuario'].apply(normalize_text)

svm_model = svm.SVC(C=1.0, kernel='linear')
# Crear el pipeline con SMOTE y SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
    ('svm', svm_model)
])

pipeline.fit(df['Historia de usuario'], df['Bien formada'])
# Guardar el modelo completo en un archivo
joblib.dump(pipeline, 'modelo_svm.joblib')
