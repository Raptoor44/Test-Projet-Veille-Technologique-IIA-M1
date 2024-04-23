import TicTacToe
import random

def test(agent, num_games):
    num_wins = 0
    for i in range(num_games):
        state = TicTacToe().board
        while not TicTacToe(state).game_over():
            if TicTacToe(state).player == 1:
                action = agent.choose_action(state, TicTacToe(state).available_moves())
            else:
                action = random.choice(TicTacToe(state).available_moves())
            state, reward = TicTacToe(state).make_move(action)
        if reward == 1:
            num_wins += 1
    return num_wins / num_games * 100