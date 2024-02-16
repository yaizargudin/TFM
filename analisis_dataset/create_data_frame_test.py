from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from tabulate import tabulate
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import spacy

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


df = pd.read_excel('datos.xlsx', index_col=None)
df.columns = ['Historia de usuario', 'Bien formada']

# Crear un histograma para visualizar la distribución
frecuencias = df['Bien formada'].value_counts()
plt.pie(frecuencias, labels=frecuencias.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución de valores')
plt.show()

## OBTENER UNA MUESTRA DE CADA CLASE PRE Y POST NORMALIZACION
# Obtén la lista única de clases
clases_unicas = df['Bien formada'].unique()

# Muestra una muestra aleatoria del DataFrame para cada clase
subset_df = pd.DataFrame()
for clase in clases_unicas:
    nueva_entrada = df[df['Bien formada'] == clase].sample(1)
    subset_df = pd.concat([subset_df, nueva_entrada])

print(tabulate(subset_df, headers='keys', tablefmt='psql'))

subset_df['Historia de usuario'] = subset_df['Historia de usuario'].apply(normalize_text)
print(tabulate(subset_df, headers='keys', tablefmt='psql'))

## fin de la obtencion de muestras

df['Historia de usuario'] = df['Historia de usuario'].apply(normalize_text)



X = df['Historia de usuario']
y = df['Bien formada']

vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('spanish'))
X_train_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, y, test_size=0.2, random_state=42)

# Aplicar SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Crear un histograma para visualizar la distribución del conjunto de entrenamineto antes de SMOTE
frecuencias = y_train.value_counts()
plt.pie(frecuencias, labels=frecuencias.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución de valores antes de SMOTE')
plt.show()

# Crear un histograma para visualizar la distribución del conjunto de entrenamineto antes de SMOTE
frecuencias = y_train_resampled.value_counts()
plt.pie(frecuencias, labels=frecuencias.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución de valores después de SMOTE')
plt.show()

train = {'Historia de usuario': X_train_resampled, 'Bien formada': y_train_resampled}
train_data = pd.DataFrame(train)
train_data.to_csv('train_data.csv', index=False)
test = {'Historia de usuario': X_test, 'Bien formada': y_test}
test_data = pd.DataFrame(test)
test_data.to_csv('test_data.csv', index=False)