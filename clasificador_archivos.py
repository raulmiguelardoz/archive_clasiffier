# -*- coding: utf-8 -*-
"""
Clasificador de archivos usando un modelo previamente entrenado.
"""

import os
import joblib
import shutil

class ClasificadorArchivos:
    def __init__(self, carpeta):
        """ Inicializa la clase con la carpeta donde están los archivos y el modelo a usar """
        self.carpeta = carpeta
        self.nombres_ficheros = []
        self.modelo = joblib.load('C:\\Users\\rmmendoza\\Desktop\\archive_clasiffier\\model\\modelo_entrenado.pkl') # Cargar el modelo en el constructor

    def extraer_nombres_ficheros(self):
        """ Extrae los nombres de los ficheros (con extensión) de la carpeta """
        if not os.path.exists(self.carpeta):
            raise FileNotFoundError(f"La carpeta {self.carpeta} no existe.")
        
        for filename in os.listdir(self.carpeta):
            file_path = os.path.join(self.carpeta, filename)
            if os.path.isfile(file_path):
                # Extraemos solo el nombre del archivo con la extensión
                nombre_archivo = filename
                self.nombres_ficheros.append(nombre_archivo)

        print(f"Nombres de ficheros extraídos: {len(self.nombres_ficheros)}")
        
        return self.nombres_ficheros

    def predecir_etiquetas(self):
        """ Predice las etiquetas para los ficheros dados """
        if self.modelo is None:
            raise ValueError("El modelo no ha sido cargado. Llama a 'cargar_modelo' primero.")
        
        return self.modelo.predict(self.nombres_ficheros)
    
    def mover_archivos_a_carpetas(self):
        """ Mueve los archivos a carpetas correspondientes según sus etiquetas """
        etiquetas_predichas = self.predecir_etiquetas()
    
        for nombre_archivo, etiqueta in zip(self.nombres_ficheros, etiquetas_predichas):
            # Crear la carpeta con el nombre de la etiqueta (si no existe)
            carpeta_destino = os.path.join(self.carpeta, etiqueta)
            if not os.path.exists(carpeta_destino):
                os.makedirs(carpeta_destino)  # Si no existe, la creamos
            
            archivo_origen = os.path.join(self.carpeta, nombre_archivo)
            archivo_destino = os.path.join(carpeta_destino, nombre_archivo)
            
            # Mover el archivo a la carpeta correspondiente
            shutil.move(archivo_origen, archivo_destino)
            print(f"Archivo '{nombre_archivo}' movido a la carpeta '{etiqueta}'.")

# Ejecutar
if __name__ == "__main__":
    # Ruta de la carpeta con los archivos
    carpeta = 'C:\\Users\\rmmendoza\\Desktop\\archive_clasiffier\\files\\files8'  # Cambia esto por la ruta real de tu carpeta

    # Crear el clasificador
    clasificador = ClasificadorArchivos(carpeta)

    try:
        # Extraer los nombres de los ficheros
        clasificador.extraer_nombres_ficheros()

        # Hacer las predicciones
        etiquetas_predichas = clasificador.predecir_etiquetas()

        # Mostrar las predicciones
        for nombre, etiqueta in zip(clasificador.nombres_ficheros, etiquetas_predichas):
            print(f"El archivo '{nombre}' fue clasificado como: {etiqueta}")

        # Mover los archivos a sus respectivas carpetas
        clasificador.mover_archivos_a_carpetas()

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
