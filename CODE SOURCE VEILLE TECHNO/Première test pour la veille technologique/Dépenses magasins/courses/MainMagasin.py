import tensorflow as tf
import numpy as np
import random

import Magasin


# Définir l'environement
class Environement:
    def __init__(self):
        self.month = 0
        self.scoreTotalGagne = 0

    def step(self, action):
        reward = 0
        for i in range(0, 12):
            magasinA = Magasin.Magasin(random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3))
            magasinA.calculate_total_score()
            magasinB = Magasin.Magasin(random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3))
            magasinB.calculate_total_score()

            if self.month % 10 == 0: #magasin 2
                magasinC = Magasin.Magasin(0.0000001, 0.0000001, 0.0000001,
                                           0.0000001, 0.0000000000001,
                                           0.0000001, 0.0000001)
            else:
                magasinC = Magasin.Magasin(0.0011, random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                           random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                           random.uniform(0.7, 1.3))

            magasinC.calculate_total_score()
            magasinD = Magasin.Magasin(random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3))
            magasinD.calculate_total_score()
            magasinE = Magasin.Magasin(random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3))
            magasinE.calculate_total_score()

            magasinF = Magasin.Magasin(random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3))
            magasinF.calculate_total_score()
            magasinG = Magasin.Magasin(random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3))
            magasinG.calculate_total_score()
            magasinH = Magasin.Magasin(random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3), random.uniform(0.7, 1.3), random.uniform(0.7, 1.3),
                                       random.uniform(0.7, 1.3))
            magasinH.calculate_total_score()

            listChoice = [magasinA.total_score, magasinB.total_score, magasinC.total_score, magasinD.total_score,
                          magasinE.total_score, magasinF.total_score, magasinG.total_score, magasinH.total_score]

            self.scoreTotalGagne += listChoice[action]
            reward += listChoice[action]

            self.month += 1

        print(f"le magasin sélectionner : {action}")
        return reward, True


# Paramètres
gamma = 0.9  # Gamma proche de 1, ça veut dire que l'agent prend en compte les récompenses futures quand c'est plus proche de 1.
epsilon = 0.3  # un épsilon proche de 1 indique que l'agent va essayer plus de tentative d'exploration que d'exploitaiton.
learning_rate = 0.9999 #Plus cette valeur est proche de 1, plus l'algorithme va prendre de l'importance au nouvelle information.
num_actions = 7  # Nombre d'actions possibles
# Entrainement avec Q-learning
num_episodes = 300

# Modèle Q
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_actions)
])

# Optimiseur
optimizer = tf.keras.optimizers.Adam(learning_rate)


# Fonction d'action epsilon-greedy
def epsilon_greedy(state):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    else:
        q_values = model.predict(np.array([state]))
        return np.argmax(q_values)


# Environement
environement = Environement()
environement.step(5)

for episode in range(num_episodes):
    state = np.array([environement.month])
    total_reward = 0

    while True:
        # Choisir une action avec l'algorithme Q-learning
        action = epsilon_greedy(state)
        # Appliquer l'action à l'environnement
        well_being, done = environement.step(action)
        # Calculer la récompense
        reward = well_being

        # Mettre à jour le modèle Q-learning
        with tf.GradientTape() as tape:
            q_values = model(np.array([state]))
            target = q_values.numpy()
            target[0, action] = reward + gamma * np.max(q_values)
            loss = tf.keras.losses.mean_squared_error(q_values, target)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# À la fin de l'entraînement, vous pouvez utiliser le modèle pour prendre des décisions.
# Par exemple, pour le budget restant de 1300 euros, utilisez model.predict(np.array([1300])) pour obtenir les Q-values,
# puis choisissez l'action avec np.argmax().
