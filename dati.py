import numpy as np
import pandas as pd
import os
import json
import pickle
import random
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Concatenate, Flatten, 
                                     Layer, RepeatVector, Subtract, LeakyReLU)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam   
from utils import *

def mean_multiply(y_true, y_pred):
    return K.mean(y_true * y_pred)

class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        # grads = K.gradients(target, wrt)
        grads = tf.gradients(target, wrt, unconnected_gradients='zero')
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

class T2V(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):  

        self.W = self.add_weight(name='W',
                    shape=(input_shape[-1], self.output_dim),
                    initializer='uniform',
                    trainable=True)            
        
        self.P = self.add_weight(name='P',
                    shape=(input_shape[1], self.output_dim),
                    initializer='uniform',
                    trainable=True)        
        self.w = self.add_weight(name='w',
                    shape=(input_shape[1], 1),
                    initializer='uniform',
                    trainable=True)        
        self.p = self.add_weight(name='p',
                    shape=(input_shape[1], 1),
                    initializer='uniform',
                    trainable=True)        
        super(T2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        
        return K.concatenate([sin_trans, original], -1)

def def_actor(env, time_emb, toward='forward', **kwargs):
    # Input state (previous or current) as condition and extract features
    in_space = env.observation_space if toward=='forward' else env.action_space
    out_space = env.action_space if toward=='forward' else env.observation_space
    state = Input(shape=in_space.shape[0])
    state_feats = Dense(kwargs['input_units'], activation='tanh')(state)
    # Input noise in the actions represented in latent space
    noise = Input(shape=kwargs['latent_dim'])
    noise_feats = Dense(kwargs['input_units'], activation='tanh')(noise)
    # Time input 
    if kwargs['add_time']:
        time = [Input(shape=(1,))] 
        time_emb_feats = time_emb(time[0])
        time_emb_feats = [Flatten()(time_emb_feats)]
    else:
        time, time_emb_feats = [], []
    # Mix the information
    concat = Concatenate()([state_feats, noise_feats] + time_emb_feats)
    hidden = Dense(kwargs['hidden_units'], activation='tanh')(concat)
    for _ in range(kwargs['num_hidden']-1):
        hidden = Dense(kwargs['hidden_units'], activation='tanh')(hidden)
    # Get the outputs
    out = Dense(out_space.shape[0], activation='tanh')(hidden)
    model = Model([state, noise] + time, out)
    return model

def def_critic(env, time_emb, toward='forward', **kwargs):
    in_space = env.action_space if toward=='forward' else env.observation_space
    # Action as input
    action_or_state = Input(shape=in_space.shape[0])
    action_or_state_feats = Dense(kwargs['input_units'], activation='tanh')(action_or_state)
    # Time input (positional encoding)
    if kwargs['add_time']:
        time = [Input(shape=(1,))] 
        time_emb_feats = time_emb(time[0])
        time_emb_feats = [Flatten()(time_emb_feats)]
    else:
        time, time_emb_feats = [], []
    # Help the critic to judge based on reward
    if kwargs['add_reward']:
        reward = [Input(shape=(1,))]
        other_feats_dim = action_or_state_feats.shape[-1]
        other_feats_dim += time_emb_feats[0].shape[-1] if kwargs['add_time'] else 0
        tile_reward = Flatten()(RepeatVector(other_feats_dim)(reward[0]))
        tile_reward = [Dense(other_feats_dim, kernel_constraint='non_neg')(tile_reward)]
    else:
        reward, tile_reward = [], []
    # Both are passed through separate layers before concatenating
    concat = Concatenate()([action_or_state_feats] + time_emb_feats + tile_reward)
    # Process the joint information
    hidden = Dense(kwargs['hidden_units'])(concat)
    hidden = LeakyReLU(0.2)(hidden)
    for _ in range(kwargs['num_hidden']-1):
        hidden = Dense(kwargs['hidden_units'])(hidden)
        hidden = LeakyReLU(0.2)(hidden)
    # Outputs single value for give state-action
    output = Dense(1, activation='elu')(hidden)
    model = Model([action_or_state] + time + reward, output)
    return model

class DATI(object):
    
    def __init__(self, env, render=None, save_dir=None):
        self.env = env 
        self.reset_seed(env.current_seed)
        self.save_dir = save_dir
        self.render = render

        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
            
        if not os.path.exists(f'{save_dir}/test_dapi_{env.tracks.name}.csv'):
            r = pd.DataFrame([], columns=['seed', 'bestDTW'])
            r.to_csv(f'{save_dir}/test_dapi_{env.tracks.name}.csv', index=False)
    
    def reset_seed(self, seed):
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def policy(self, prev_xy_action, step=0):
        tf_state = K.expand_dims(prev_xy_action, 0)
        noise_t =  K.expand_dims(self.noise_obj(), 0)  
        time = K.expand_dims(np.array([step]), 0)
        time = [time] if self.add_time else []
        action = self.actor([tf_state, noise_t] + time)
        speed, angle = tf.squeeze(action)
        return np.array([speed, angle]), noise_t

    def inv_policy(self, action, noise_t, step=0):
        tf_action = K.expand_dims(action, 0)
        time = K.expand_dims(np.array([step]), 0)
        time = [time] if self.add_time else []
        inv_action = self.inv_actor([tf_action, noise_t] + time)
        return np.squeeze(inv_action)
        
    def build_half_cycle(self, actor, critic, inv_actor, toward='forward', **kwargs):
        lr_actor = kwargs['lr_actor'] if toward == 'forward' else kwargs['lr_inv_actor']
        lr_critic = kwargs['lr_critic'] if toward == 'forward' else kwargs['lr_inv_critic']
        lr_l1 = kwargs['lr_l1']
        lmbd_grad = kwargs['lambda_grad_pen']
        lmbd_l1 = kwargs['lambda_l1']

        a_opt = Adam(learning_rate=lr_actor, beta_1=0.5, beta_2=0.9)
        c_opt = Adam(learning_rate=lr_critic, beta_1=0.5, beta_2=0.9)
        l1_opt = Adam(learning_rate=lr_l1)

        in_space = self.env.observation_space if toward=='forward' else self.env.action_space
        out_space = self.env.action_space if toward=='forward' else self.env.observation_space
        prev_xy_action = Input(in_space.shape[0])
        noise = Input(shape=self.noise_obj.latent_dim)
        action_real = Input(out_space.shape[0])
        action_interp = Input(out_space.shape[0])
        time = [Input(shape=(1,))] if self.add_time else []
        reward = [Input(shape=(1,))] if self.add_reward else []
        extra = time + reward

        # Output: D(G(z))-D(x), norm ==(y_true: nones, ones)==> Loss: D(G(z))-D(x)+lmbd_grad*(norm-1)**2
        actor.trainable = False
        critic.trainable = True
        action = actor([prev_xy_action, noise] + time)
        sub = Subtract()([critic([action] + extra), critic([action_real] + extra)])
        norm = GradNorm()([critic([action_interp] + extra), action_interp])
        critic_gan = Model([prev_xy_action, noise, action_real, action_interp] + extra, [sub, norm]) 
        loss_sub, loss_norm = mean_multiply, 'mse'
        critic_gan.compile(optimizer=c_opt, loss=[loss_sub, loss_norm], loss_weights=[1.0, lmbd_grad])

        # Output: D(G(z)) ==(y_true: -1*ones)==> Loss: (-1) * D(G(z))
        actor.trainable = True
        critic.trainable = False
        action = actor([prev_xy_action, noise] + time)
        value_action = critic([action] + extra)
        actor_gan = Model([prev_xy_action, noise] + extra, value_action)
        actor_gan.compile(optimizer=a_opt, loss=mean_multiply)

        # Output F(G(z|x)), G(z|x) ==(y_true: x, action_real)==> Loss:  lmbd_l1 * L1 + lmbd_l1 * L1
        inv_actor.trainable = True
        inv_action = inv_actor([action, noise] + time)
        output_l1 = [inv_action, action] if toward == 'forward' else [inv_action,]
        half_cycle_l1 = Model([prev_xy_action, noise] + time, output_l1)
        half_cycle_l1.compile(optimizer=l1_opt, loss=['mae', 'mae'], loss_weights=[lmbd_l1, lmbd_l1])

        return critic_gan, actor_gan, half_cycle_l1
    
    def build(self, **kwargs):
        with open(f'{self.save_dir}/args_{self.env.tracks.name}.json', 'w') as foo:
            json.dump(kwargs, foo)

        self.add_reward = kwargs['add_reward']
        self.add_time = kwargs['add_time']

        if kwargs['latent_type'] == 'Ornstein-Uhlenbeck':
            self.noise_obj = OrnsteinUhlenbeckNoise(
                mean=0.0, 
                std_dev= kwargs['latent_std_dev'], 
                dt=self.env.tracks.dt, 
                latent_dim=kwargs['latent_dim']
            )
        else:
            self.noise_obj = GaussianNoise(
                mean=0.0, 
                std_dev= kwargs['latent_std_dev'], 
                latent_dim=kwargs['latent_dim']
            )
        
        self.replay_buffer = ReplayBuffer(
            self.noise_obj,
            buffer_capacity = 50000,
            batch_size = kwargs['batch_size']
        )
        # Initial warm up to show the networks good behavior
        self.replay_buffer.warmup(self.env)

        # Define actor and critic networks
        self.time_emb = T2V(kwargs['time_emb_dim'])
        self.actor = def_actor(self.env, self.time_emb, **kwargs) 
        self.critic = def_critic(self.env, self.time_emb, **kwargs) 
        self.inv_actor = def_actor(self.env, self.time_emb, **kwargs) 
        self.inv_critic = def_critic(self.env, self.time_emb, **kwargs)

        # Define the forward model
        model_forward = self.build_half_cycle(
            self.actor, 
            self.critic, 
            self.inv_actor, 
            toward='forward',
            **kwargs
        )
        # Define the backward model
        model_backward = self.build_half_cycle(
            self.inv_actor, 
            self.inv_critic, 
            self.actor, 
            toward='backward',
            **kwargs
        )
        self.critic_gan = model_forward[0]
        self.actor_gan = model_forward[1]
        self.half_cycle_l1 = model_forward[2]
        self.inv_critic_gan = model_backward[0]
        self.inv_actor_gan = model_backward[1]
        self.inv_half_cycle_l1 = model_backward[2]

    def train_on_batch(self, **kwargs):
        nbatch = kwargs['batch_size']
        ncritic =kwargs['critic_steps_per_actor']

        # Sample experiences from the replay buffer
        batch = self.replay_buffer.sample()
        time = [batch['time']] if self.add_time else []
        reward = [batch['reward']] if self.add_reward else []

        eps = np.random.uniform(0, 1, (nbatch, 1))
        action_interp = eps * batch['action_real'] + (1-eps) * batch['action']
        prev_xy_interp = eps * batch['inv_action'] + (1-eps) * batch['prev_xy_action']

        # Update the critics
        for _ in range(ncritic):
            critic_gan_losses = self.critic_gan.train_on_batch(
                [
                    batch['prev_xy_action'], 
                    batch['noise'], 
                    batch['action_real'], 
                    action_interp
                ] + time + reward,
                2 * [np.ones((nbatch, 1))]
            )
            inv_critic_gan_losses = self.inv_critic_gan.train_on_batch(
                [
                    batch['action'], 
                    batch['noise'], 
                    batch['prev_xy_action'], 
                    prev_xy_interp
                ] + time + reward,
                2 * [np.ones((nbatch, 1))]
            )
        
        # Update the actors
        actor_gan_losses = self.actor_gan.train_on_batch(
            [
                batch['prev_xy_action'], 
                batch['noise']
            ] + time + reward,
            [
                -1.0 * np.ones((nbatch, 1)), 
            ]
        )
        inv_actor_gan_losses = self.inv_actor_gan.train_on_batch(
            [
                batch['action'], 
                batch['noise']
            ] + time + reward,
            [
                -1.0 * np.ones((nbatch, 1)), 
            ]
        )
        half_cycle_l1_losses = self.half_cycle_l1.train_on_batch(
            [
                batch['prev_xy_action'],
                batch['noise']
            ] + time,
            [
                batch['prev_xy_action'],
                batch['action_real']
            ]
            
        )
        inv_half_cycle_l1_losses = self.inv_half_cycle_l1.train_on_batch(
            [
                batch['action'],
                batch['noise']
            ] + time,
            [
                batch['action'],
            ]
            
        )
        total_loss = 0.0
        total_loss += critic_gan_losses[0]
        total_loss += inv_critic_gan_losses[0]
        total_loss += actor_gan_losses
        total_loss += inv_actor_gan_losses
        total_loss += half_cycle_l1_losses[0]
        total_loss += inv_half_cycle_l1_losses
        return {
            'critic_sub': critic_gan_losses[1],
            'icritic_sub': inv_critic_gan_losses[1],
            'critic_grad': critic_gan_losses[2],
            'icritic_grad': inv_critic_gan_losses[2],
            'actor_gan': actor_gan_losses,
            'iactor_gan': inv_actor_gan_losses,
            'cyc_l1': half_cycle_l1_losses[1],
            'icyc_l1': inv_half_cycle_l1_losses,
            'actor_l1': half_cycle_l1_losses[2],
            'total_loss': total_loss
        } 

    def fit(self, **kwargs):

        self.build(**kwargs)
        logger = Logger(self.env.num_episodes, self.env.timesteps)
        early_stop = EarlyStopCheckpoint(patience=10)

        for episode in range(self.env.num_episodes):
            self.env.reset()
            self.noise_obj.reset()
            while True:
                losses = self.train_on_batch(**kwargs) 
                step = self.env.step_counter-1
                prev_xy_action = self.env.tracks.scale_position(self.env.prev_xy_action)
                action, noise = self.policy(prev_xy_action, step=step)
                inv_action = self.inv_policy(action, noise, step=step)
                # Get reward and next state from the action taken
                _, reward, done, _ = self.env.step(action)
                logger.log(losses, reward, self.env.dtw_dist)
                velocity_now = self.env.tracks.velocity_at_t(step * self.env.tracks.dt)
                action_real = self.env.tracks.scale_velocity(velocity_now)
                # Record experience
                self.replay_buffer.record(
                    time = step,
                    reward = reward,
                    noise = noise,
                    prev_xy_action = prev_xy_action,
                    action = action,
                    action_real = action_real,
                    inv_action = inv_action
                )
                if done: break
                if self.render: self.env.render()
            if self.render: self.env.close()
            if early_stop.monitor(logger, self).now: break

        return logger
    
    def evaluate(self, test_runs=10, seed_train=0, seed_test=0, last_episode=1):
        # Test in multiple runs, choose best performance
        self.reset_seed(seed_test)
        self.env.seed(int(seed_test))
        dtw_dist = np.empty((test_runs,))
        all_xy_actions = []
        all_states = []
        # Bring best model after early stopping
        best = f'weights-{self.env.tracks.name}_episode-{last_episode}_seed-{seed_train}.h5'
        if os.path.exists(f'{self.save_dir}/{best}'): 
            self.actor.load_weights(f'{self.save_dir}/{best}')

        for k in range(test_runs):
            prev_xy_action = self.env.tracks.scale_position(self.env.reset())
            self.noise_obj.reset()
            while True:
                step = self.env.step_counter-1
                action, _ = self.policy(prev_xy_action, step=step)
                _, reward, done, _ = self.env.step(action)
                prev_xy_action = self.env.tracks.scale_position(self.env.prev_xy_action)
                print(f'Step: {step},  Reward: {reward}, dtw_dist: {self.env.dtw_dist:.2f}')
                if done: break
                if self.render: self.env.render()
            if self.render: self.env.close()

            dtw_dist[k] = self.env.dtw_dist
            all_xy_actions.append(self.env.history_xy_action[:self.env.step_counter])
            all_states.append(self.env.history_state[:self.env.step_counter])

        pickle.dump(
            [all_states, all_xy_actions], 
            open(f'{self.save_dir}/rollouts_{self.env.tracks.name}_{seed_test}.pkl', 'wb')
        )
        r = pd.DataFrame([], columns=['seed', 'bestDTW'])
        r.loc[0] = [seed_test, np.min(dtw_dist,0)]
        r.to_csv(f'{self.save_dir}/test_dapi_{self.env.tracks.name}.csv', 
                index=False, header=False, mode='a')