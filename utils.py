import time
import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt
from visdom import Visdom

class ReplayBuffer:
    def __init__(self, noise_obj, buffer_capacity=100000, batch_size=64, num_states=2, num_actions=2):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        self.latent_dim = noise_obj.latent_dim

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.time_buffer = np.zeros((self.buffer_capacity, 1))
        self.prev_xy_action_buffer = np.zeros((self.buffer_capacity, num_actions)) 
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.action_real_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.inv_action_buffer = np.zeros((self.buffer_capacity, num_states))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype='int32')
        self.noise_buffer = np.zeros((self.buffer_capacity, self.latent_dim))

    def record(self, **obs):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.time_buffer[index] = obs['time']
        self.prev_xy_action_buffer[index] = obs['prev_xy_action']
        self.action_buffer[index] = obs['action']
        self.action_real_buffer[index] = obs['action_real']
        self.inv_action_buffer[index] = obs['inv_action']
        self.reward_buffer[index] = obs['reward']
        self.noise_buffer[index] = obs['noise']

        self.buffer_counter += 1

    def sample(self):
        indices = np.random.choice(self.buffer_counter, self.batch_size)
        return {
            'time': self.time_buffer[indices],
            'reward': self.reward_buffer[indices],
            'noise': self.noise_buffer[indices],
            'prev_xy_action': self.prev_xy_action_buffer[indices],
            'action_real': self.action_real_buffer[indices],
            'action': self.action_buffer[indices],
            'inv_action': self.inv_action_buffer[indices],
        }
    
    def warmup(self, env, plot_samples=False, dangle=np.radians(5)):

        step_counter = np.random.choice(env.timesteps-1, (self.batch_size,), replace=False)
        all_positions = env.tracks.get_positions()
        all_velocities = env.tracks.get_velocities()
        # Add initial correct steering to the batch
        prev_xy_action = env.tracks.scale_position(all_positions[0])
        action_real = env.tracks.scale_velocity(all_velocities[0])
        self.record(
            time = 0,
            prev_xy_action = prev_xy_action,
            action = action_real,
            action_real = action_real,
            inv_action = prev_xy_action,
            reward = 1,
            noise = np.zeros((self.latent_dim,))
        )
        
        for k in range(self.batch_size-1):
            time = step_counter[k]
            state_prev = all_positions[time]
            state_now = all_positions[time+1]
            delta_state = state_now - state_prev
            speed_true, angle_true = velocity_now = all_velocities[time]

            # Randomly sample an action with true speed
            theta_a = np.random.normal(angle_true, dangle)
            unscaled_action = np.array([speed_true, theta_a])
            action = env.tracks.scale_velocity(unscaled_action)
            action_real = env.tracks.scale_velocity(velocity_now)
            xy_action = env.tracks.update_position(state_prev, unscaled_action)
            approx_prev_state = delta_state - xy_action
            inv_action = env.tracks.scale_position(approx_prev_state)
            # Get reward if meaningful action
            reward = 1 if abs(theta_a - angle_true) < dangle else -1

            self.record(
                time = time,
                prev_xy_action = env.tracks.scale_position(state_prev),
                action = action,
                action_real = action_real,
                inv_action = inv_action,
                reward = reward,
                noise = np.random.normal((self.latent_dim,))
            )

            if plot_samples:
                color = 'g' if reward == 1 else 'r'
                plt.scatter(*state_prev, color=color)
                plt.scatter(*state_now, color=color)
        
        if plot_samples: plt.show()

class OrnsteinUhlenbeckNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, latent_dim=64, x_initial=None):
        self.latent_dim = latent_dim
        self.theta = theta
        self.mean = mean * np.ones(latent_dim)
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt +     
             self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
        return self.x_prev

class GaussianNoise:
    def __init__(self, mean, std_dev, latent_dim=64, x_initial=None):
        self.mean = mean * np.ones(latent_dim)
        self.std_dev = std_dev
        self.latent_dim = latent_dim
        self.x_initial = x_initial
        self.reset()
    
    def __call__(self):
        x = np.random.normal(self.mean, self.std_dev, size=self.latent_dim)
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros(self.latent_dim)
        return self.x_prev

class Logger(object):

    def __init__(self, n_episodes, batches_episode) -> None:
        self.viz = Visdom()
        self.n_episodes = n_episodes
        self.batches_episode = batches_episode
        self.batch = 1
        self.episode = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.total_loss = 0
    
    def log(self, losses, reward, dtw_dist):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        progress = int(100 * self.batch / self.batches_episode)
        sys.stdout.write(f'\rEpisode {self.episode}/{self.n_episodes} [{progress:02}%] -- ')

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name]
            else:
                self.losses[loss_name] += losses[loss_name]

            if (i+1) == len(losses.keys()):
                sys.stdout.write(f'{loss_name}: {self.losses[loss_name]/self.batch:.4f} -- ')
            else:
                sys.stdout.write(f'{loss_name}: {self.losses[loss_name]/self.batch:.4f} | ')

        sys.stdout.write(f'reward: {reward} -- ')
        sys.stdout.write(f'dtw_dist: {dtw_dist:.2f} -- ')
        
        batches_done = self.batches_episode *(self.episode - 1) + self.batch
        batches_left = self.batches_episode *(self.n_episodes - self.episode) + self.batches_episode - self.batch 
        sys.stdout.write(f'ETA: {datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)}')

        # End of episode
        if (self.batch % self.batches_episode) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                loss_per_sample = loss / self.batch
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(
                        X=np.array([self.episode]), 
                        Y=np.array([loss_per_sample]), 
                        opts={'xlabel': 'episodes', 'ylabel': loss_name, 'title': loss_name}
                    )
                else:
                    self.viz.line(
                        X=np.array([self.episode]), 
                        Y=np.array([loss_per_sample]), 
                        win=self.loss_windows[loss_name], 
                        update='append'
                    )
                # Reset losses for next episode
                self.losses[loss_name] = 0.0
                # Accumulate the total loss 
                if 'total_loss' in loss_name:
                    self.total_loss += loss_per_sample

            self.episode += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

class EarlyStopCheckpoint(object):

    def __init__(self, patience=0):
        self.patience = patience
        self.best_weights = None
        self.wait = 0
        self.best_loss = np.Inf
        self.now = False
    
    def monitor(self, logger, model):
        loss = logger.total_loss
        if loss < self.best_loss:
            self.best_loss = loss 
            self.wait = 0
            self.best_weights = model.actor.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                episode = logger.episode
                seed = model.env.current_seed
                name = model.env.tracks.name
                params = '_'.join([
                    f'weights-{name}',
                    f'episode-{episode}',
                    f'seed-{seed}'
                ])
                # Get the best weights so far
                model.actor.set_weights(self.best_weights)
                model.actor.save_weights(
                    f'{model.save_dir}/{params}.h5'
                )
                self.now = True
        return self



    