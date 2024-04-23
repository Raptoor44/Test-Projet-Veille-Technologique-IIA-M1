import configFile
import tensorflow as tf
import numpy as np

# Charger le modèle depuis le fichier
model = tf.keras.models.load_model(configFile.MODEL_PATH)

# Utiliser le modèle pour prédire le sens d'une nouvelle phrase
nouvelle_phrase = "La souris est mangée par le chat."
word_to_index = {word: idx for idx, word in enumerate(set(nouvelle_phrase.split()))}
nouvelle_indices = [word_to_index[word] for word in nouvelle_phrase.split() if word in word_to_index]
prediction = model.predict(np.array([nouvelle_indices]))

print(f"Prédiction pour la phrase '{nouvelle_phrase}': {prediction}")
