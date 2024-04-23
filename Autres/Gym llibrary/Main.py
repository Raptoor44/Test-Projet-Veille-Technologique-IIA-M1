# q learning with table
import numpy as np
import Game as Game
states_n = 16
actions_n = 4
Q = np.zeros([states_n, actions_n])

# Set learning parameters
lr = .85
y = .99
num_episodes = 1000
cumul_reward_list = []
actions_list = []
states_list = []
game = Game(4, 4, 0) # 0.1 chance to go left or right instead of asked direction
for i in range(num_episodes):
    actions = []
    s = game.reset()
    states = [s]
    cumul_reward = 0
    d = False
    while True:
        # on choisit une action aléatoire avec une certaine probabilité, qui décroit
        # TODO : simplifier ça (pas clair)
        Q2 = Q[s,:] + np.random.randn(1, actions_n)*(1. / (i +1))
        a = np.argmax(Q2)
        s1, reward, d, _ = game.move(a)
        Q[s, a] = Q[s, a] + lr*(reward + y * np.max(Q[s1,:]) - Q[s, a]) # Fonction de mise à jour de la Q-table
        cumul_reward += reward
        s = s1
        actions.append(a)
        states.append(s)
        if d == True:
            break
    states_list.append(states)
    actions_list.append(actions)
    cumul_reward_list.append(cumul_reward)

print("Score over time: " +  str(sum(cumul_reward_list[-100:])/100.0))

game.reset()
game.print()