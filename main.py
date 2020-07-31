from game import Game
from dqn import DQN
import tensorflow as tf
import datetime
import numpy as np
tf.compat.v1.enable_eager_execution()


gamma = 0.99
copy_step = 25
num_states = 8
num_actions = 18
hidden_units = [200, 200]
max_experiences = 10000
min_experiences = 100
batch_size = 256
lr = 1e-2
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/dqn/' + current_time
summary_writer = tf.contrib.summary.create_file_writer(log_dir)

TrainNet = DQN(num_states, num_actions, hidden_units, gamma,
               max_experiences, min_experiences, batch_size, lr)
TargetNet = DQN(num_states, num_actions, hidden_units, gamma,
                max_experiences, min_experiences, batch_size, lr)

# TrainNet.load_model('models/successful_go_outside/train/model_900.h5')
# TargetNet.load_model('models/successful_go_outside/target/model_900.h5')

N = 50000
total_rewards = np.empty(N)
epsilon = 0.99
decay = 0.999
min_epsilon = 0.1

f = open('logs.txt', 'a')

game = Game()

for n in range(N):
    epsilon = max(min_epsilon, epsilon * decay)

    total_reward, losses = game.game_loop(TrainNet, TargetNet, epsilon, copy_step)
    total_rewards[n] = total_reward
    avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
    with summary_writer.as_default():
        tf.compat.v1.summary.scalar('episode reward', total_reward)
        tf.compat.v1.summary.scalar('running avg reward(100)', avg_rewards)
        tf.compat.v1.summary.scalar('average loss)', losses)
    if n % 100 == 0:
        print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                "episode loss: ", losses)
        TrainNet.save_model('models/train/', n)
        TargetNet.save_model('models/target/', n)
    print("avg reward for last 100 episodes:", avg_rewards)
    f.write(str(avg_rewards)+"\n")
f.close()

