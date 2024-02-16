from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

nlp = spacy.load("es_core_news_sm")

# Función de normalización de texto
def normalize_text(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar caracteres de puntuación
    texto = re.sub(r'[^\w\s]', '', texto)
    # Procesar el texto con spaCy
    doc = nlp(texto)
    # Obtener lemas de cada token en el texto
    lemmas = [token.lemma_ for token in doc]
    # Unir tokens nuevamente en texto
    texto = ' '.join(lemmas)
    return texto


# Cargar el conjunto de datos
df = pd.read_excel('datos.xlsx', index_col=None)
df.columns = ['Historia de usuario', 'Bien formada']
df['Historia de usuario'] = df['Historia de usuario'].apply(normalize_text)

X = np.array(df['Historia de usuario'])
y = np.array(df['Bien formada'])


# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el pipeline con SMOTE y SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
    ('svm', SVC())
])
# Definir la cuadrícula de parámetros a buscar
param_grid = {
    'svm__C': [0.1, 1, 10, 100],  # Valores del parámetro de regularización C
    'svm__kernel': ['linear', 'rbf','poly']  # Tipos de kernel a probar
}

# Inicializar GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Realizar la búsqueda de la cuadrícula
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros:", grid_search.best_params_)

# Hacer predicciones en el conjunto de prueba
y_pred = grid_search.predict(X_test)


# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión usando seaborn
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",annot_kws={"size": 60})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()