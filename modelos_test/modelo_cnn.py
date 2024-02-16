import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re

from sklearn.utils import resample

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


# Convertir etiquetas a codificación one-hot
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['Bien formada'])
one_hot_labels = to_categorical(encoded_labels)

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['Historia de usuario'], one_hot_labels, test_size=0.2, random_state=42)

# Tokenización y secuenciación de texto
max_words = 1000  # Número máximo de palabras a considerar
max_len = 200     # Longitud máxima de las secuencias
X_train_resampled, y_train_resampled = resample(X_train, y_train, random_state=42)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_resampled)

X_train_seq = tokenizer.texts_to_sequences(X_train_resampled)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Creación del modelo CNN
embedding_dim = 50
vocab_size = max_words

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(one_hot_labels.shape[1], activation='softmax'))

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_train_pad, y_train_resampled , epochs=50, batch_size=32, validation_data=(X_test_pad, y_test))

# Evaluación del modelo
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
report = classification_report(y_test_classes, y_pred_classes)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Visualizar la matriz de confusión usando seaborn
plt.figure(figsize=(3, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",xticklabels=['0', '1','3','4'],yticklabels=['0', '1','3','4'],annot_kws={"size": 60})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()