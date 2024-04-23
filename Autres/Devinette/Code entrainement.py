import numpy as np
import tensorflow as tf

# Environnement de devinette simple
class DevinetteEnvironment:
    def __init__(self):
        self.state_space_size = 10  # Par exemple, 10 devinettes différentes
        self.action_space_size = 2  # Deux actions possibles: deviner 'Vrai' ou 'Faux'
        self.current_state = np.random.randint(0, self.state_space_size)  # État initial aléatoire

    def reset(self):
        self.current_state = np.random.randint(0, self.state_space_size)

    def step(self, action):
        # Récompense basée sur la réponse correcte ou incorrecte
        reward = 1 if action == self.current_state % 2 else -1
        done = False  # Dans cet exemple, l'épisode ne se termine jamais
        return self.current_state, reward, done

# Modèle du réseau neuronal pour Q-learning
class QLearningModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(QLearningModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')
        self.compile(optimizer='adam', loss='mean_squared_error')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)


# Agent Q-learning
class QLearningAgent:
    def __init__(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.model = QLearningModel(action_space_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def select_action(self, state, exploration_rate=0.1):
        state = np.asarray(state)  # Assurez-vous que state est un tableau NumPy
        if state is None:
            return 0  # Ou toute autre action par défaut, car le state est invalide
        if np.random.rand() < exploration_rate:
            return np.random.choice(self.action_space_size)
        else:
            q_values = self.model.predict(state.reshape(1, -1))  # Utilisez reshape pour vous assurer d'un tableau 1D
            return np.argmax(q_values)

    def train(self, state, action, reward, next_state):
        target = reward + 0.99 * np.max(self.model.predict(next_state[np.newaxis, :]))
        with tf.GradientTape() as tape:
            q_values = self.model(state[np.newaxis, :])
            action_one_hot = tf.one_hot(action, self.action_space_size)
            selected_q = tf.reduce_sum(q_values * action_one_hot, axis=1)
            loss = tf.keras.losses.mean_squared_error(target, selected_q)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

# Fonction d'entraînement
def train_agent(agent, env, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state)
            total_reward += reward
            state = next_state
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Création de l'environnement et de l'agent
env = DevinetteEnvironment()
agent = QLearningAgent(state_space_size=env.state_space_size, action_space_size=env.action_space_size)


# Entraînement de l'agent
train_agent(agent, env)
