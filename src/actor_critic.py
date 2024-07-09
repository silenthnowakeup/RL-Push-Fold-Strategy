import tensorflow as tf


class ActorCritic(tf.keras.Model):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(512, activation="relu")  # Increased size
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(256, activation="relu")  # Increased size
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.critic = tf.keras.layers.Dense(1, activation=None)
        self.actor = tf.keras.layers.Dense(action_dim, activation='softmax')

    def call(self, input_data):
        x = self.fc1(input_data)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return critic, actor
