import os
import cv2
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ruta relativa a la carpeta de imágenes (si el código y las imágenes están en el mismo directorio)
images_path = '/Users/giovannyhidalgo/Desktop/archive/images/images1/'

# Función para cargar y redimensionar imágenes
def load_and_resize_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        resized_image = cv2.resize(image, (28, 28))
        return resized_image
    else:
        return None

X_data = []
Y_data = []

# Generador de datos de imágenes con configuración optimizada
datagen = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

# Cargar imágenes y generar datos de entrenamiento
for filename in os.listdir(images_path):
    image_path = os.path.join(images_path, filename)
    image = load_and_resize_image(image_path)
    if image is not None:
        X_data.append(image)
        img_name = filename.split('.')[0]
        Y_data.append(img_name)

        # Generar 100 imágenes adicionales para el Pokémon actual usando el generador de datos
        for _ in range(100):
            random_img = datagen.random_transform(image)
            X_data.append(random_img)
            Y_data.append(img_name)

if not X_data or not Y_data:
    raise ValueError("No se encontraron imágenes válidas en el directorio.")

X_data = np.array(X_data).astype('float32') / 255.0

encoder = LabelEncoder()
Y_data = encoder.fit_transform(Y_data)
Y_data = to_categorical(Y_data)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Arquitectura del modelo (más profunda para mejorar el rendimiento)
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
# Agregar más capas convolucionales y de normalización aquí
# ...
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenar el modelo con configuración adicional
model.fit(datagen.flow(X_train, Y_train, batch_size=192),
          epochs=5,
          steps_per_epoch=X_train.shape[0] // 192,
          validation_data=(X_test, Y_test),
          callbacks=[EarlyStopping(monitor='loss', min_delta=1e-10, patience=20, verbose=1),
                     ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, verbose=1),
                     ModelCheckpoint(filepath='weightssss.h5', monitor='loss',
                                     save_best_only=True, verbose=1)])

# Guardar el modelo
model.save('model.h5')


# Evaluar el modelo en datos de prueba y entrenamiento
evaluation_test = model.evaluate(X_test, Y_test)
print("Evaluación en datos de prueba:")
print("Pérdida:", evaluation_test[0])
print("Precisión:", evaluation_test[1])

evaluation_train = model.evaluate(X_train, Y_train)
print("Evaluación en datos de entrenamiento:")
print("Pérdida:", evaluation_train[0])
print("Precisión:", evaluation_train[1])

# Función para cargar, redimensionar y preprocesar una nueva imagen
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        resized_image = cv2.resize(image, (28, 28))
        preprocessed_image = resized_image.astype('float32') / 255.0
        return preprocessed_image
    else:
        return None

# Función para realizar una predicción de Pokémon a partir de una nueva imagen
def predict_pokemon(image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    if preprocessed_image is not None:
        preprocessed_image = preprocessed_image.reshape(1, 28, 28, 3)  # Agregar dimensión del lote (batch) a la imagen
        prediction_probabilities = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(prediction_probabilities)
        predicted_pokemon = encoder.inverse_transform([predicted_class_index])[0]
        return predicted_pokemon
    else:
        return "No se pudo cargar la imagen."

# Ejemplo de cómo usar la función de predicción con una nueva imagen
new_image_path = '/Users/giovannyhidalgo/Desktop/archive/images/images1/pikachu.png'
predicted_pokemon = predict_pokemon(new_image_path)
print("Predicción de Pokémon:", predicted_pokemon)
