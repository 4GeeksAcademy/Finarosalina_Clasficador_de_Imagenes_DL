import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import set_random_seed
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import os
import shutil

# Ruta donde están todas las imágenes extraídas del zip
original_dataset_dir = '/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/data/raw/train'

# Directorio base donde se crearán los datasets
base_dir = '/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/data/processed'
os.makedirs(base_dir, exist_ok=True)

# Subdirectorios de train y validation
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

for subdir in [train_dir, val_dir]:
    os.makedirs(os.path.join(subdir, 'cats'), exist_ok=True)
    os.makedirs(os.path.join(subdir, 'dogs'), exist_ok=True)

# Obtiene todos los nombres de archivos
filenames = os.listdir(original_dataset_dir)
cat_filenames = [f for f in filenames if f.startswith('cat')]
dog_filenames = [f for f in filenames if f.startswith('dog')]

# Divide en entrenamiento y validación
train_cats, val_cats = train_test_split(cat_filenames, test_size=0.2, random_state=42)
train_dogs, val_dogs = train_test_split(dog_filenames, test_size=0.2, random_state=42)

# Función para mover archivos
def mover_archivos(file_list, target_dir):
    for fname in file_list:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(target_dir, fname)
        shutil.move(src, dst)

# Mueve imágenes de gatos
mover_archivos(train_cats, os.path.join(train_dir, 'cats'))
mover_archivos(val_cats, os.path.join(val_dir, 'cats'))

# Mueve imágenes de perros
mover_archivos(train_dogs, os.path.join(train_dir, 'dogs'))
mover_archivos(val_dogs, os.path.join(val_dir, 'dogs'))




import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
import numpy as np

train_dir = '/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/data/processed/train'

# Función para cargar y redimensionar imágenes
def load_and_resize_image(img_path, target_size=(200, 200)):
    img = image.load_img(img_path, target_size=target_size)
    return image.img_to_array(img)

# Cargar las primeras 9 imágenes de gatos y perros
cat_dir = os.path.join(train_dir, 'cats') 
dog_dir = os.path.join(train_dir, 'dogs') 
cat_filenames = os.listdir(cat_dir)
dog_filenames = os.listdir(dog_dir)

# Seleccionar las primeras 9 imágenes
cat_images = [load_and_resize_image(os.path.join(cat_dir, f)) for f in cat_filenames[:9]]
dog_images = [load_and_resize_image(os.path.join(dog_dir, f)) for f in dog_filenames[:9]]

# Crear un gráfico para mostrar las imágenes
fig, axes = plt.subplots(3, 6, figsize=(12, 8))
axes = axes.ravel()

for i in range(9):
    # Mostrar imágenes de gatos
    axes[i].imshow(cat_images[i].astype('uint8'))
    axes[i].axis('off')
    
    # Mostrar imágenes de perros
    axes[i + 9].imshow(dog_images[i].astype('uint8'))
    axes[i + 9].axis('off')

plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear el generador de imágenes para datos de entrenamiento con escala de grises
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizar las imágenes
    shear_range=0.2,  # Cortes aleatorios
    zoom_range=0.2,   # Zoom aleatorio
    horizontal_flip=True  # Flip horizontal aleatorio
)

# Crear el generador para los datos de validación
validation_datagen = ImageDataGenerator(rescale=1./255)

# Directorios de entrenamiento y validación
train_dir = '/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/data/processed/train'
validation_dir = '/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/data/processed/validation'

# Cargar imágenes de entrenamiento y validación en escala de grises
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),  # Redimensionar las imágenes
    batch_size=32,  # Tamaño de lote
    class_mode='binary',  # Dos clases: perro y gato
    color_mode='grayscale'  # Convertir las imágenes a escala de grises
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(200, 200),  # Redimensionar las imágenes
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'  # Convertir las imágenes a escala de grises
)


from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

# Configurar la semilla para reproducibilidad
set_random_seed(42)

# Modelo secuencial para imágenes en escala de grises
model = Sequential([
    # Definimos explícitamente la forma de entrada utilizando Input
    Input(shape=(200, 200, 1)),  # Imagen en escala de grises de 200x200
    Flatten(),  # Aplana la imagen a un vector unidimensional
    Dense(128, activation="relu"),  # Capa oculta con 128 neuronas
    Dense(1, activation="sigmoid")  # Capa de salida para clasificación binaria
])

# Mostrar resumen del modelo
model.summary()

# Compilación del modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Entrenamiento del modelo
history = model.fit(
    train_generator,  # Generador de entrenamiento
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Número de pasos por cada época
    epochs=10,  # Número de épocas
    validation_data=validation_generator,  # Generador de validación
    validation_steps=validation_generator.samples // validation_generator.batch_size  # Pasos de validación
)


import matplotlib.pyplot as plt

# Graficar precisión
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Graficar pérdida
plt.figure()
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()


from tensorflow.keras.layers import Dense, Flatten, Input, Dropout

model = Sequential([
    Input(shape=(200, 200, 1)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=7,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Definir el modelo
model_cnn = Sequential([
    Input(shape=(200, 200, 1)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilación del modelo
model_cnn.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.summary()

# Callback para guardar el mejor modelo
checkpoint_callback = ModelCheckpoint(
    '/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/models/best_model_cnn.h5',  # Ruta donde se guardará el mejor modelo
    monitor='val_loss',  # Monitorea la pérdida de validación
    save_best_only=True,  # Solo guarda el mejor modelo
    mode='min',  # Buscando minimizar la pérdida
    verbose=1  # Para ver mensajes cuando se guarda el modelo
)

# Callback para EarlyStopping
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Espera 3 épocas sin mejora antes de detenerse
    restore_best_weights=True  # Restaurar los pesos del mejor modelo
)

# Entrenamiento del modelo con callbacks
history = model_cnn.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,  # Reducción de epochs
    callbacks=[checkpoint_callback, early_stop_callback]
)

# Guardar el modelo al final del entrenamiento
model_cnn.save('/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/models/cnn_cats_dogs_final_v1.keras')


model_cnn.save("/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/models/cnn_cats_dogs_final_v1.keras")


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.legend()
plt.show()


from tensorflow.keras.models import load_model

# Cargar el modelo guardado
final_model = load_model('/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/models/cnn_cats_dogs_final_v1.keras')

# Evaluar el modelo en el conjunto de validación
val_loss, val_accuracy = final_model.evaluate(validation_generator)
print(f"Loss en validación: {val_loss}")
print(f"Precisión en validación: {val_accuracy}")


import nbformat


notebook_file = '/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/src/explore.ipynb'
python_file = '/workspaces/Finarosalina_Clasficador_de_Imagenes_DL/src/app.py'


with open(notebook_file, 'r') as f:
    notebook_content = nbformat.read(f, as_version=4)


code_cells = [cell['source'] for cell in notebook_content['cells'] if cell['cell_type'] == 'code']


with open(python_file, 'w') as f:
    for code in code_cells:
        f.write(code + '\n\n')



