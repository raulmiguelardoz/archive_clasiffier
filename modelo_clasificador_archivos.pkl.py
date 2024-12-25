import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def entrenar_modelo(fichero_pruebas, nombre_modelo='modelo_entrenado.pkl'):
    """Entrena el modelo usando los datos del fichero de pruebas y guarda el modelo entrenado"""
    # Leer el fichero de pruebas
    df = pd.read_csv(fichero_pruebas, sep=';')
    print(df.columns)

    # Verificar que el fichero tenga las columnas correctas
    if 'Nombre' not in df.columns or 'Etiqueta' not in df.columns:
        raise ValueError("El fichero de pruebas debe tener columnas 'Nombre' y 'Etiqueta'.")
    
    # Extraemos las características (nombres de los ficheros) y las etiquetas
    nombres_ficheros = df['Nombre'].tolist()
    etiquetas = df['Etiqueta'].tolist()
      
    # Crear el modelo usando un pipeline con TfidfVectorizer y Naive Bayes
    modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    # Entrenar el modelo
    modelo.fit(nombres_ficheros, etiquetas)
    
    # Guardar el modelo entrenado
    joblib.dump(modelo, nombre_modelo)  # Guardar el modelo con el nombre que des
    
    print(f"Modelo entrenado y guardado en {nombre_modelo}.")

def cargar_modelo(nombre_modelo='modelo_entrenado.pkl'):
    """Carga el modelo previamente entrenado"""
    return joblib.load(nombre_modelo)
    
def predecir_etiqueta(modelo, nombres_ficheros):
    """Predice las etiquetas para los ficheros dados"""
    return modelo.predict(nombres_ficheros)

#-----------------------------------------------------------------------------------------------------------

# Execute
if __name__ == "__main__":
    # Ruta del fichero de prueba
    fichero_pruebas = 'C:\\Users\\rmmendoza\\Desktop\\archive_clasiffier\\model\\training_dataset.csv'  # Cambia esto por tu archivo de prueba
    
    # Entrenar el modelo
    entrenar_modelo(fichero_pruebas, 'modelo_entrenado.pkl')

    # Cargar el modelo entrenado
    modelo = cargar_modelo('modelo_entrenado.pkl')
    
    # Cargar el conjunto de pruebas para generar la matriz de confusión
    df_test = pd.read_csv(fichero_pruebas, sep=';')
    nombres_ficheros_test = df_test['Nombre'].tolist()  # Aquí obtienes los nombres de los archivos
    etiquetas_test = df_test['Etiqueta'].tolist()  # Aquí obtienes las etiquetas reales

    # Realizar las predicciones
    etiquetas_predichas = predecir_etiqueta(modelo, nombres_ficheros_test)
    
    # Matriz de confusión
    cm = confusion_matrix(etiquetas_test, etiquetas_predichas)
    print("Matriz de Confusión:")
    print(cm)

    # Visualizar la matriz de confusión como un heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()

    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(etiquetas_test, etiquetas_predichas))
