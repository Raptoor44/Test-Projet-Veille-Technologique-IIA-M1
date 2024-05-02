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
        if action == 0:  # Acheter
            self.current_index[company_index] += 1
        elif action == 1:  # Vendre
            if self.current_index[company_index] > 0:
                self.current_index[company_index] -= 1
        # Ne rien faire si action == 2
        pass

    def get_reward(self, company_index):
        # Calculer la récompense pour l'action de l'agent sur une entreprise donnée
        current_price = self.companies_data[company_index][self.current_index[company_index]]
        previous_price = self.companies_data[company_index][self.current_index[company_index] - 1]
        reward = current_price - previous_price
        return reward

# Définir l'agent utilisant un réseau de neurones pour l'apprentissage
class DQNAgent:
    def __init__(self, state_size, action_sizes):
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.num_companies = len(action_sizes)
        self.models = {company: self._build_model(company_state_size, action_size) for company, company_state_size, action_size in zip(action_sizes.keys(), [state_size] * self.num_companies, action_sizes.values())}
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Taux de remise
        self.epsilon = 1.0  # Exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _build_model(self, state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='huber')
        return model

    def remember(self, state, action, reward, next_state, done):
        memory_entry = (state, action, reward, next_state, done)
        self.memory.append(memory_entry)

    def act(self, state, company_index):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_sizes[company_index])
            print(f"Action choisie (exploration) pour {company_index}:", action)
            return action
        act_values = self.models[company_index].predict(state)[0]
        action = np.argmax(act_values)
        print(f"Action choisie (exploitation) pour {company_index}:", action)
        return action

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            company_index = np.argmax(state)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.models[company_index].predict(next_state)[0])
            target_f = self.models[company_index].predict(state)
            target_f[0]

# Charger les données historiques des actions (remplacez 'data.csv' par votre fichier)
companies_data = np.loadtxt('data.csv', delimiter=',')

# Définir les paramètres de l'environnement et de l'agent
state_size = 1  # Dimension de l'état (prix actuel de l'action)
action_sizes = {  # Actions possibles par entreprise
    "Total": 3,  # Acheter, vendre, conserver
    # ... (définir les actions pour d'autres entreprises)
}
batch_size = 32  # Taille du lot pour l'entraînement de l'agent
num_episodes = 10  # Nombre d'épisodes d'entraînement

# Créer l'environnement et l'agent
env = Environment(companies_data)
agent = DQNAgent(state_size, action_sizes)

# Entraînement de l'agent
total_reward = 0  # Initialiser le gain total
for episode in range(num_episodes):
    state = env.get_state().reshape(1, state_size)  # Obtenir l'état initial
    print("Début de l'épisode", episode)
    for time in range(100):  # Limite de 100 pas de temps par épisode
        print("Pas de temps", time)
        company_index = np.argmax(state)  # Identifier l'entreprise en cours
        action = agent.act(state, company_index)  # L'agent prend une action
        env.take_action(action, company_index)  # L'action est appliquée à l'environnement
        next_state = env.get_state().reshape(1, state_size)  # Obtenir le nouvel état
        reward = env.get_reward(company_index)  # Obtenir la récompense
        total_reward += reward  # Ajouter la récompense à la somme totale
        print(f"Récompense pour l'action ({company_index}): {reward}")
        done = False  # Indicateur pour déterminer si l'épisode est terminé
        agent.remember(state, action, reward, next_state, done)
        state = next_state  # Mettre à jour l'état actuel
        if done:
            print(f"Épisode terminé, gain total de bourse: {total_reward}")
            total_reward = 0  # Réinitialiser le gain total pour le prochain épisode
            break
        if len(agent.memory) > batch_size:
            agent.train(batch_size)

# Évaluation de l'agent (non implémenté ici)
# ... (Ajouter une logique d'évaluation pour tester les performances de l'agent)