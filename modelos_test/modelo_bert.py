import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch
import unidecode
import string
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

# Dividir el conjunto de datos en entrenamiento y prueba
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Definir el modelo BERT para clasificación
modelo = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=5)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# Tokenizar el conjunto de datos
class DatasetClasificacion(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        texto = normalize_text(str(self.dataframe.iloc[index]['Historia de usuario']))
        bien_formada = self.dataframe.iloc[index]['Bien formada']

        codificacion = self.tokenizer.encode_plus(
            texto,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'texto': texto,
            'input_ids': codificacion['input_ids'].flatten(),
            'attention_mask': codificacion['attention_mask'].flatten(),
            'bien_formada': torch.tensor(bien_formada, dtype=torch.long)
        }


max_len = 32
conjunto_entrenamiento = DatasetClasificacion(train_df, tokenizer, max_len)
conjunto_prueba = DatasetClasificacion(test_df, tokenizer, max_len)

# Configurar el entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo.to(device)

batch_size = 4
entrenamiento_dataloader = DataLoader(conjunto_entrenamiento, batch_size=batch_size, shuffle=True)
prueba_dataloader = DataLoader(conjunto_prueba, batch_size=batch_size, shuffle=False)

# Configurar el optimizador y la función de pérdida
optimizador = AdamW(modelo.parameters(), lr=2e-5)
num_epochs = 3

# Entrenar el modelo
for epoch in range(num_epochs):
    modelo.train()
    total_loss = 0

    for batch in tqdm(entrenamiento_dataloader, desc=f'Época {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        bien_formada = batch['bien_formada'].to(device)

        optimizador.zero_grad()

        outputs = modelo(input_ids, attention_mask=attention_mask, labels=bien_formada)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizador.step()

    print(f'Pérdida de entrenamiento promedio: {total_loss / len(entrenamiento_dataloader)}')

# Evaluar el modelo en el conjunto de prueba
modelo.eval()
predicciones, etiquetas_reales = [], []

with torch.no_grad():
    for batch in tqdm(prueba_dataloader, desc='Evaluación'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        bien_formada = batch['bien_formada'].to(device)

        outputs = modelo(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        predicciones.extend(logits.argmax(dim=1).cpu().numpy())
        etiquetas_reales.extend(bien_formada.cpu().numpy())

# Calcular métricas de evaluación
from sklearn.metrics import accuracy_score, classification_report

# Medir la precisión del modelo
precision = accuracy_score(etiquetas_reales, predicciones)
print(f'Precisión del modelo: {precision:.2f}')

print('\nInforme de clasificación:')
print(classification_report(etiquetas_reales, predicciones))

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(etiquetas_reales, predicciones)

# Visualizar la matriz de confusión usando seaborn
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",annot_kws={"size": 60})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()