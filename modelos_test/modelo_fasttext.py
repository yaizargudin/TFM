import pandas as pd
from sklearn.model_selection import train_test_split
import fasttext
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
import spacy

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

def guardar_datos_formato_fasttext(data_frame, file_name):
    # Guardar los datos en un formato que pueda ser utilizado por FastText
    with open(file_name, 'w', encoding='utf-8') as f:
        for ind in data_frame.index:
            f.write(
                f'__label__' + str(data_frame['Bien formada'][ind]) + ' ' + normalize_text(data_frame['Historia de usuario'][ind]) + '\n')


df = pd.read_excel('datos.xlsx', index_col=None)
df.columns = ['Historia de usuario', 'Bien formada']


##Entrenamiento
# Dividir los datos en conjunto de entrenamiento y prueba
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
guardar_datos_formato_fasttext(train_data, 'train.txt')
modelo = fasttext.train_supervised(input='train.txt', lr=1.0, epoch=50, wordNgrams=5)
# Hacer predicciones en el conjunto de prueba
predicciones = [modelo.predict(texto)[0][0].replace('__label__', '') for texto in test_data['Historia de usuario']]
etiquetas_reales = [str(etiquetas) for etiquetas in test_data['Bien formada']]

# Medir la precisión del modelo
precision = accuracy_score(etiquetas_reales, predicciones)
print(f'Precisión del modelo: {precision:.2f}')

# Mostrar el informe de clasificación
print('\nInforme de clasificación:')
print(classification_report(etiquetas_reales, predicciones))

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(etiquetas_reales, predicciones)

# Visualizar la matriz de confusión usando seaborn
plt.figure(figsize=(3, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",annot_kws={"size": 60})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()