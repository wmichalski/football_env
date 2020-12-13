from game import Game
from dqn import DQN
from player import Player
import tensorflow as tf
import datetime
import numpy as np

tf.compat.v1.enable_eager_execution()
tf.config.optimizer.set_jit(True)

from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float32')
# mixed_precision.set_policy(policy)


gamma = 0.99
copy_step = 100
num_states = 12
num_actions = 9
hidden_units = [200, 200]
max_experiences = 1000
min_experiences = 100
batch_size = 32
lr = 1e-3
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/dqn/' + current_time
summary_writer = tf.compat.v2.summary.create_file_writer(logdir=log_dir)

TrainNet1 = DQN(num_states, num_actions, hidden_units, gamma,
               max_experiences, min_experiences, batch_size, lr)
TargetNet1 = DQN(num_states, num_actions, hidden_units, gamma,
                max_experiences, min_experiences, batch_size, lr)

TrainNet2 = DQN(num_states, num_actions, hidden_units, gamma,
               max_experiences, min_experiences, batch_size, lr)
TargetNet2 = DQN(num_states, num_actions, hidden_units, gamma,
                max_experiences, min_experiences, batch_size, lr)

player1 = Player()
player1.Train = TrainNet1
player1.Target = TargetNet1

player2 = Player()
player2.Train = TrainNet2
player2.Target = TargetNet2

players = [player1, player2]

N = 50000
total_rewards = np.empty(N)
epsilon = 0.9
decay = 0.999
min_epsilon = 0.1

f = open('logs.txt', 'a', buffering=1024)

game = Game(players)

for n in range(N):
    epsilon = max(min_epsilon, epsilon * decay)
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if n % 5 == 0:
        player1.Target.copy_weights(player1.Train)
        player2.Target.copy_weights(player2.Train)

    game.game_loop(epsilon, copy_step)

    # total_reward, losses = game.game_loop(epsilon, copy_step)
    # total_rewards[n] = total_reward
    # avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
    # with summary_writer.as_default():
    #     tf.compat.v1.summary.scalar('episode reward', total_reward)
    #     tf.compat.v1.summary.scalar('running avg reward(100)', avg_rewards)
    #     tf.compat.v1.summary.scalar('average loss)', losses)
    if n % 100 == 0:
        print(n, " epsilon: ", epsilon)
        # print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
        #         "episode loss: ", losses)
        player1.Train.save_model('models/p1/train/', n)
        player1.Target.save_model('models/p1/target/', n)
        player2.Train.save_model('models/p2/train/', n)
        player2.Target.save_model('models/p2/target/', n)
        f.flush()
    # print("avg reward for last 100 episodes:", avg_rewards)
    # f.write(str(avg_rewards)+"\n")
f.close()
