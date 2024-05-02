import numpy as np
import tensorflow as tf
from collections import deque
import random

# Définir l'environnement de simulation (données historiques sur Total)
class Environment:
    def __init__(self, companies_data):
        self.companies_data = companies_data
        self.num_companies = len(companies_data)
        self.current_index = np.zeros(self.num_companies, dtype=int)

    def get_state(self):
        # Retourner l'état actuel de l'environnement pour chaque entreprise
        return [self.companies_data[i][self.current_index[i]] for i in range(self.num_companies)]

    def take_action(self, action, company_index):
        # Exécuter l'action de l'agent sur l'environnement pour une entreprise donnée
        pass  # À remplacer par la logique appropriée pour l'entreprise spécifique

    def get_reward(self, company_index):
        # Calculer la récompense pour l'action de l'agent sur une entreprise donnée
        return np.random.randn()  # Récompense aléatoire pour l'exemple

# Définir l'agent utilisant un réseau de neurones pour l'apprentissage
class DQNAgent:
    def __init__(self, state_size, action_size):
        # Initialiser les paramètres de l'agent et construire le modèle
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Taux de remise
        self.epsilon = 1.0  # Exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _build_model(self):
        # Construction du modèle neuronal pour l'apprentissage
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        # Ajouter l'expérience à la mémoire de l'agent
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Prendre une action basée sur l'état actuel de l'environnement
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)  # Exploration
            print("Action choisie (exploration):", action)
            return action
        act_values = self.model.predict(state)
        action = np.argmax(act_values[0])  # Exploitation
        print("Action choisie (exploitation):", action)
        return action

    def train(self, batch_size):
        # Entraîner l'agent en utilisant l'algorithme DQN
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

# Paramètres de l'agent et de l'environnement
state_size = 1  # Dimension de l'état (p. ex. prix actuel de l'action)
action_size = 3  # Nombre d'actions possibles (p. ex. acheter, vendre, ne rien faire)
batch_size = 32  # Taille du lot pour l'entraînement de l'agent
total_reward = 0  # Initialiser le gain total

# Créer l'environnement
companies_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Exemple de données pour trois entreprises
env = Environment(companies_data)

agent = DQNAgent(state_size, action_size)
# Entraînement de l'agent
num_episodes = 10  # Nombre d'épisodes d'entraînement
for episode in range(num_episodes):
    state = np.array(env.get_state()).reshape(1, state_size)  # Obtenir l'état initial de l'environnement
    print("Début de l'épisode", episode)
    for time in range(100):  # Limite de 100 pas de temps par épisode
        print("Pas de temps", time)
        action = agent.act(state)  # L'agent prend une action
        print("Action choisie par l'agent:", action)
        env.take_action(action,
                        company_index=0)  # L'action est appliquée à l'environnement (exemple avec company_index=0)
        next_state = np.array(env.get_state()).reshape(1,
                                                       state_size)  # Obtenir le nouvel état de l'environnement
        reward = env.get_reward(
            company_index=0)  # Obtenir la récompense pour l'action (exemple avec company_index=0)
        total_reward += reward  # Ajouter la récompense à la somme totale
        print("Récompense pour l'action:", reward)
        done = False  # Indicateur pour déterminer si l'épisode est terminé
        agent.remember(state, action, reward, next_state, done)
        state = next_state  # Mettre à jour l'état actuel
        if done:
            print("Épisode terminé, gain total de bourse:", total_reward)
            total_reward = 0  # Réinitialiser le gain total pour le prochain épisode
            break
        if len(agent.memory) > batch_size:
            agent.train(batch_size)