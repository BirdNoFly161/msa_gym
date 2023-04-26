from MultipleSequenceAlignmentEnv import * 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Dense, Flatten,MaxPooling2D
from tensorflow.keras.optimizers import Adam

from rl.agents import CEMAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = MultipleSequenceAlignmentEnv(['FGKGKC','FGKFGK','GKGKC','KFKC','GKG'])
states_shape = (1,) + env.observation_space.shape
actions = env.n_actions
print(states_shape, actions)
# cnn = to input the sequences
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=states_shape))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(actions, activation='linear'))
model.build()
print(model.summary())

agent = CEMAgent(
    model=model,
    memory=SequentialMemory(limit=5000, window_length=1),
    nb_actions=actions,
    nb_steps_warmup=10,
)

agent.compile()
agent.fit(env, nb_steps=5000, visualize=False, verbose=1)
results = agent.test(env, nb_episodes=5, visualize=True)
print(results.history['episode_reward'])
                 
                 







if __name__ == "__main__":

   
    obs = env.reset()
    for _ in range (15):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
 
    env.print_mat_string_alignment()
