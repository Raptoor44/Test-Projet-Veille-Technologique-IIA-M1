import tensorflow as tf
import numpy as np

# Définir l'environnement
class Environment:
    def __init__(self, prices, well_being_scores, unit_obligatory):
        self.prices = prices
        self.well_being_scores = well_being_scores
        self.unit_obligatory = unit_obligatory
        self.budget = 1300
        self.month = 0

    def step(self, action):
        price = self.prices[action] * np.random.uniform(0.7, 1.3)  # Variation de prix
        well_being = self.well_being_scores[action]
        self.budget -= price

        print(f"Month: {self.month}, Length of unit_obligatory: {len(self.unit_obligatory)}")

        # Gérer les contraintes de l'unité obligatoire
        if len(self.unit_obligatory) > self.month:
            if self.unit_obligatory[self.month] > 0:
                self.unit_obligatory[self.month] -= 1

        self.month += 1
        if self.month > 12:
            done = True
        else:
            done = False

        return well_being, done

# Paramètres
gamma = 0.9
epsilon = 0.1
learning_rate = 0.01
num_actions = 7  # Nombre d'actions possibles

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

# Environnement
prices = [50, 12, 20, 100, 30, 10, 5]
well_being_scores = [10, 5, 15, 10, 7, 10, 20]
unit_obligatory = [1, 0, 0, 4, 0, 4, 0]
env = Environment(prices, well_being_scores, unit_obligatory)

# Entraînement avec Q-learning
num_episodes = 30

for episode in range(num_episodes):
    state = np.array([env.budget])
    total_reward = 0

    while True:
        # Choisir une action avec l'algorithme Q-learning
        action = epsilon_greedy(state)



        # Appliquer l'action à l'environnement
        well_being, done = env.step(action)

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