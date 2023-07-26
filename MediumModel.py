import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt  # Importar explícitamente el módulo Qt
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Ruta relativa a la carpeta de imágenes (si el código y las imágenes están en el mismo directorio)
images_path = '/Users/giovannyhidalgo/Desktop/archive/images/images/'

# Función para cargar, redimensionar y preprocesar una nueva imagen
def cargar_y_preprocesar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is not None:
        imagen_redimensionada = cv2.resize(imagen, (28, 28))
        imagen_preprocesada = imagen_redimensionada.astype('float32') / 255.0
        return imagen_preprocesada
    else:
        return None

# Función para realizar una predicción de Pokémon a partir de una nueva imagen
def predecir_pokemon(ruta_imagen):
    imagen_preprocesada = cargar_y_preprocesar_imagen(ruta_imagen)
    if imagen_preprocesada is not None:
        imagen_preprocesada = imagen_preprocesada.reshape(1, 28, 28, 3)  # Agregar dimensión del lote (batch) a la imagen
        probabilidades_prediccion = modelo.predict(imagen_preprocesada)
        indice_clase_predicha = np.argmax(probabilidades_prediccion)
        pokemon_predicho = encoder.inverse_transform([indice_clase_predicha])[0]
        return pokemon_predicho
    else:
        return "No se pudo cargar la imagen."

# Cargar el modelo previamente entrenado
modelo = load_model('/Users/giovannyhidalgo/Desktop/model.h5')
clases_encoder = np.load('/Users/giovannyhidalgo/Desktop/encoder_classes.npy')
encoder = LabelEncoder()
encoder.classes_ = clases_encoder

class PredictorPokemon(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Predicción de Pokémon')
        self.setGeometry(100, 100, 400, 200)

        self.etiqueta_imagen = QLabel(self)
        self.etiqueta_imagen.setAlignment(Qt.AlignCenter)  # Corregir aquí
        self.etiqueta_imagen.setFixedSize(150, 150)

        self.etiqueta_resultado = QLabel(self)
        self.etiqueta_resultado.setAlignment(Qt.AlignCenter)  # Corregir aquí

        self.boton_cargar = QPushButton('Cargar imagen', self)
        self.boton_cargar.clicked.connect(self.cargar_imagen)

        self.boton_predecir = QPushButton('Predecir', self)
        self.boton_predecir.clicked.connect(self.predecir_pokemon)

        layout = QVBoxLayout()
        layout.addWidget(self.etiqueta_imagen)
        layout.addWidget(self.etiqueta_resultado)
        layout.addWidget(self.boton_cargar)
        layout.addWidget(self.boton_predecir)

        self.setLayout(layout)

    def cargar_imagen(self):
        opciones = QFileDialog.Options()
        opciones |= QFileDialog.ReadOnly
        ruta_imagen, _ = QFileDialog.getOpenFileName(self, "Cargar imagen", "", "Imágenes (*.png *.jpg *.jpeg)", options=opciones)

        if ruta_imagen:
            self.ruta_imagen = ruta_imagen
            self.mostrar_imagen(ruta_imagen)

    def mostrar_imagen(self, ruta_imagen):
        pixmap = QPixmap(ruta_imagen)
        self.etiqueta_imagen.setPixmap(pixmap.scaled(self.etiqueta_imagen.size(), Qt.KeepAspectRatio))

    def predecir_pokemon(self):
        if hasattr(self, 'ruta_imagen'):
            pokemon_predicho = predecir_pokemon(self.ruta_imagen)
            self.mostrar_resultado(pokemon_predicho)

    def mostrar_resultado(self, nombre_pokemon):
        self.etiqueta_resultado.setText(f'Pokémon: {nombre_pokemon}')

if __name__ == '__main__':
    app = QApplication([])
    ventana = PredictorPokemon()
    ventana.show()
    app.exec_()
