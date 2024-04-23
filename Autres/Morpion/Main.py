import train
import test

# Train the Q-learning agent
agent = train.train(num_episodes=100000, alpha=0.5, epsilon=0.1, discount_factor=1.0)

# Test the Q-learning agent
win_percentage = test.test(agent, num_games=1000)
print("Win percentage: {:.2f}%".format(win_percentage))