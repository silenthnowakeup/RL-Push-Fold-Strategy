import numpy as np
import tensorflow as tf
from agent import Agent
from environment import PokerEnvironment
from config import ACTOR_MODEL_PATH, CRITIC_MODEL_PATH, TRAIN_EPISODES, GAMMA


def train_agent(episodes=TRAIN_EPISODES):
    env = PokerEnvironment()
    agent = Agent(gamma=GAMMA)
    rewards = []
    avg_rewards = []

    summary_writer = tf.summary.create_file_writer('logs/')

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        # Alternate starting player
        if episode % 2 == 1:
            state[-1] = 1  # Start with BB

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_rewards.append(avg_reward)
            print(f'Episode {episode + 1}/{episodes}, Average Reward: {avg_reward:.2f}')

    agent.save_model(ACTOR_MODEL_PATH, CRITIC_MODEL_PATH)

    return avg_rewards

if __name__ == "__main__":
    train_agent()


