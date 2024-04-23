import random
import TicTacToe


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount_factor):
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    def get_Q_value(self, state, action):
        state_key = tuple(map(tuple, state))  # Convert NumPy array to tuple
        if (state_key, action) not in self.Q:
            self.Q[(state_key, action)] = 0.0
        return self.Q[(state_key, action)]

    def choose_action(self, state, available_moves):
        state_key = tuple(map(tuple, state))  # Convert NumPy array to tuple
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        else:
            Q_values = [self.get_Q_value(state_key, action) for action in available_moves]
            max_Q = max(Q_values)
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
                i = random.choice(best_moves)
            else:
                i = Q_values.index(max_Q)
            return available_moves[i]

    def update_Q_value(self, state, action, reward, next_state):
        state_key = tuple(map(tuple, state))  # Convert NumPy array to tuple
        next_state_key = tuple(map(tuple, next_state))  # Convert NumPy array to tuple

        next_Q_values = [self.get_Q_value(next_state_key, next_action) for next_action in TicTacToe(next_state).available_moves()]
        max_next_Q = max(next_Q_values) if next_Q_values else 0.0
        self.Q[(state_key, action)] += self.alpha * (reward + self.discount_factor * max_next_Q - self.Q[(state_key, action)])
