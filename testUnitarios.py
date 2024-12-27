# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 08:01:26 2024

@author: rmmendoza
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
from clasificador_archivos import ClasificadorArchivos
 # Asumiendo que tu archivo se llama ClasificadorArchivos.py

class TestClasificadorArchivos(unittest.TestCase):
    def setUp(self):
        """ Configuración inicial antes de cada prueba """
        self.carpeta_prueba = "test_files"
        self.modelo_mock = MagicMock()
        self.modelo_mock.predict = MagicMock(return_value=["etiqueta1", "etiqueta2"])
        
        # Crear carpeta de prueba
        os.makedirs(self.carpeta_prueba, exist_ok=True)
        # Crear archivos ficticios
        self.archivo1 = os.path.join(self.carpeta_prueba, "archivo1.txt")
        self.archivo2 = os.path.join(self.carpeta_prueba, "archivo2.csv")
        with open(self.archivo1, "w") as f:
            f.write("contenido archivo 1")
        with open(self.archivo2, "w") as f:
            f.write("contenido archivo 2")
        
        # Instancia de ClasificadorArchivos
        self.clasificador = ClasificadorArchivos(self.carpeta_prueba)
        self.clasificador.modelo = self.modelo_mock  # Reemplazar el modelo real por el mock

    def tearDown(self):
        """ Limpieza después de cada prueba """
        shutil.rmtree(self.carpeta_prueba, ignore_errors=True)
    
    def test_extraer_nombres_ficheros(self):
        """ Verifica que se extraen correctamente los nombres de los ficheros """
        nombres = self.clasificador.extraer_nombres_ficheros()
        self.assertIn("archivo1.txt", nombres)
        self.assertIn("archivo2.csv", nombres)
        self.assertEqual(len(nombres), 2)

    def test_cargar_modelo(self):
        """ Verifica que el modelo se carga adecuadamente """
        with patch('joblib.load', return_value=self.modelo_mock) as mock_load:
            clasificador = ClasificadorArchivos(self.carpeta_prueba)
            mock_load.assert_called_once_with('C:\\Users\\rmmendoza\\Desktop\\archive_clasiffier\\model\\modelo_entrenado.pkl')

    def test_predecir_etiquetas(self):
        """ Verifica que las predicciones se generan correctamente """
        self.clasificador.extraer_nombres_ficheros()
        etiquetas = self.clasificador.predecir_etiquetas()
        self.assertEqual(etiquetas, ["etiqueta1", "etiqueta2"])
        self.modelo_mock.predict.assert_called_once_with(self.clasificador.nombres_ficheros)

    def test_mover_archivos_a_carpetas(self):
        """ Verifica que los archivos se mueven a las carpetas correspondientes """
        self.clasificador.extraer_nombres_ficheros()
        self.clasificador.mover_archivos_a_carpetas()
        
        # Verificar que las carpetas fueron creadas
        carpeta_etiqueta1 = os.path.join(self.carpeta_prueba, "etiqueta1")
        carpeta_etiqueta2 = os.path.join(self.carpeta_prueba, "etiqueta2")
        self.assertTrue(os.path.exists(carpeta_etiqueta1))
        self.assertTrue(os.path.exists(carpeta_etiqueta2))
        
        # Verificar que los archivos fueron movidos
        archivo1_destino = os.path.join(carpeta_etiqueta1, "archivo1.txt")
        archivo2_destino = os.path.join(carpeta_etiqueta2, "archivo2.csv")
        self.assertTrue(os.path.exists(archivo1_destino))
        self.assertTrue(os.path.exists(archivo2_destino))

if __name__ == '__main__':
    unittest.main()
