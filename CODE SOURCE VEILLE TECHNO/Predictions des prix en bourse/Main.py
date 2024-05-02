import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import math
import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3  # Sit, Buy, Sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # Load pre-trained model if evaluation mode, otherwise create a new model
        self.model = load_model(model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))  # Use Adam optimizer
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # Choose random action
        options = self.model.predict(state)
        return np.argmax(options[0])  # Choose action with highest predicted value

    def expReplay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)  # Sample a mini-batch from memory
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train the model on the mini-batch
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay epsilon


# Function to format price display
def formatPrice(n):
    return ("-Rs." if n < 0 else "Rs.") + "{0:.2f}".format(abs(n))


# Function to get stock data from CSV
def getStockDataVec(key):
    vec = []
    lines = open(key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))  # Extract closing price
    return vec


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Function to get state from data
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else [-data[0]] * -d + data[0:t + 1]  # Pad with initial price if needed
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))  # Normalize price changes
    return np.array([res])


# Main part of the script
stock_name = input("Enter stock_name, window_size, Episode_count")
window_size = int(input())
episode_count = int(input())
agent = Agent(window_size, False, "Total Agent")
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        if action == 1:  # Buy action
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0:  # Sell action
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
    if e % 10 == 0:
        agent.model.save(str(e))  # Save the model every 10 episodes


