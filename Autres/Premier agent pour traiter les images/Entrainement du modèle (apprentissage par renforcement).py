import tensorflow as tf
import numpy as np
import configApprentissageParRenforcement

# Chargement des données MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Création du modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle pour l'apprentissage supervisé
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Environnement pour l'apprentissage par renforcement (exemple simplifié)
class RLEnvironment:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.current_index = 0

    def reset(self):
        self.current_index = 0

    def step(self, action):
        # Récompense en fonction de la précision de la prédiction
        true_label = self.labels[self.current_index]
        predicted_label = np.argmax(model.predict(np.expand_dims(self.data[self.current_index], axis=0)))

        reward = 1 if predicted_label == true_label else -1

        # Passage à l'exemple suivant
        self.current_index += 1

        # Retourne l'observation actuelle, la récompense et une indication si l'épisode est terminé
        observation = self.data[self.current_index]
        done = self.current_index == len(self.data) - 1
        return observation, reward, done

# Création de l'environnement RL avec les données de test
rl_env = RLEnvironment(x_test, y_test)

# Exemple d'utilisation de l'environnement pour un épisode
rl_env.reset()
total_reward = 0
done = False

# Initialisation de la première observation
observation = rl_env.data[rl_env.current_index]

while not done:
    observation, reward, done = rl_env.step(model.predict(np.expand_dims(observation, axis=0)))
    total_reward += reward

print("Récompense totale de l'épisode :", total_reward)


# Entraînement du modèle pour l'apprentissage supervisé
model.fit(x_train, y_train, epochs=5)
model.save(configApprentissageParRenforcement.MODEL_PATH)
