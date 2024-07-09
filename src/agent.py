import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from actor_critic import ActorCritic

class Agent:
    def __init__(self, action_dim=2, gamma=0.99):
        self.gamma = gamma
        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.actor_critic = ActorCritic(action_dim)

    def get_action(self, state):
        state = np.array([state], dtype=np.float32)
        _, action_probabilities = self.actor_critic(state)
        action_probabilities = tf.nn.softmax(action_probabilities)
        action_probabilities = action_probabilities.numpy()[0]
        dist = tfp.distributions.Categorical(probs=action_probabilities)
        action = dist.sample().numpy()
        return action

    def actor_loss(self, prob, action, td):
        prob = tf.nn.softmax(prob)
        dist = tfp.distributions.Categorical(probs=prob)
        log_prob = dist.log_prob(action)
        loss = -log_prob * tf.stop_gradient(td)
        return loss

    def learn(self, state, action, reward, next_state, done):
        state = np.array([state], dtype=np.float32)
        next_state = np.array([next_state], dtype=np.float32)

        with tf.GradientTape() as tape:
            value, action_probabilities = self.actor_critic(state)
            next_value, _ = self.actor_critic(next_state)
            td_target = reward + self.gamma * next_value * (1 - int(done))
            td_error = td_target - value
            actor_loss = self.actor_loss(action_probabilities, action, td_error)
            critic_loss = tf.square(td_error)
            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.actor_critic.trainable_variables))
        return total_loss

    def save_model(self, actor_path, critic_path):
        self.actor_critic.save_weights(actor_path)
        self.actor_critic.save_weights(critic_path)

    def load_model(self, actor_path, critic_path):
        self.actor_critic.load_weights(actor_path)
        self.actor_critic.load_weights(critic_path)

