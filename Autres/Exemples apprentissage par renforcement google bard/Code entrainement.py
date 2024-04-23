import configFile
import tensorflow as tf
import numpy as np

# Définir l'environnement (la phrase à comprendre)
phrase = "Le chat mange la souris."
vocabulaire = set(phrase.split())
vocab_size = len(vocabulaire)

# Mapper chaque mot à un indice
word_to_index = {word: idx for idx, word in enumerate(vocabulaire)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Fonction pour convertir une phrase en indices
def phrase_to_indices(phrase, word_to_index):
    return [word_to_index[word] for word in phrase.split() if word in word_to_index]

# Définir le modèle
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=8),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Définir les paramètres de l'apprentissage par renforcement
epochs = 100
batch_size = 10

# Simuler un environnement d'apprentissage par renforcement
for epoch in range(epochs):
    indices = phrase_to_indices(phrase, word_to_index)
    state = np.zeros((1, len(phrase.split())))  # État initial
    total_reward = 0

    # Étape de collecte de données
    # Étape de collecte de données
    for t in range(len(phrase.split())):
        action_prob = model.predict(state)

        # Normaliser les probabilités pour s'assurer que la somme est égale à 1
        action_prob = action_prob / np.sum(action_prob)

        action = np.argmax(action_prob)

        # Appliquer l'action et obtenir la récompense (à définir selon le contexte)
        reward = 0.9 if index_to_word[action] == 'chat' else 0.1

        # Mettre à jour l'état et la récompense totale
        state[0, t] = action
        total_reward += reward

    # Entraîner le modèle avec la récompense totale
    model.fit(np.array([indices]), np.array([total_reward]), epochs=1, batch_size=batch_size)

# Utiliser le modèle pour prédire le sens d'une nouvelle phrase
nouvelle_phrase = "La souris est mangée par le chat."
nouvelle_indices = phrase_to_indices(nouvelle_phrase, word_to_index)
prediction = model.predict(np.array([nouvelle_indices]))
print(f"Prédiction pour la phrase '{nouvelle_phrase}': {prediction}")

# Sauvegarder le modèle
model.save(configFile.MODEL_PATH)
