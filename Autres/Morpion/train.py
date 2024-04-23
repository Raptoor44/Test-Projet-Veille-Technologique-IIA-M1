import QLearningAgent
import TicTacToe

def train(num_episodes, alpha, epsilon, discount_factor):
    agent = QLearningAgent.QLearningAgent(alpha, epsilon, discount_factor)
    game =  TicTacToe.TicTacToe()
    for i in range(num_episodes):
        state = game.board
        while not TicTacToe.TicTacToe().game_over:
            available_moves = game.available_moves()
            action = agent.choose_action(state, available_moves)
            next_state, reward = game.make_move(action)
            agent.update_Q_value(state, action, reward, next_state)
            state = next_state
    return agent
