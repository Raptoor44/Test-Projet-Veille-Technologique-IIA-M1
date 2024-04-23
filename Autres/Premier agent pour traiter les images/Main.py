import tensorflow as tf
import numpy as np
from PIL import Image
import configApprentissageSupervisée

# Load the pre-trained model
model = tf.keras.models.load_model(configApprentissageSupervisée.MODEL_PATH)

# Charger l'image que vous avez choisie
chosen_image_path = "C:\\Users\\Joachim Ecole\\Pictures\\Données de tests TensorFlow\\1.png"
chosen_image = Image.open(chosen_image_path).convert("L")  # Convertir en niveaux de gris si nécessaire
chosen_image = chosen_image.resize((28, 28))  # Redimensionner l'image à la taille attendue par le modèle
chosen_image_array = np.array(chosen_image) / 255.0  # Normaliser les valeurs des pixels

# Afficher l'image (facultatif)
chosen_image.show()

# Ajouter une dimension supplémentaire pour correspondre à la forme attendue par le modèle
input_image = np.expand_dims(chosen_image_array, axis=0)

# Faire une prédiction avec le modèle
predictions = model.predict(input_image)

# Obtenir l'indice de la classe avec la probabilité la plus élevée
predicted_class = np.argmax(predictions)

# Afficher la prédiction
print("Classe prédite :", predicted_class)
