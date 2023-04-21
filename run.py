import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tracks_env
from dati import DATI

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes of training')
    parser.add_argument('--timesteps', type=int, default=200, help='number of timesteps per episode')
    parser.add_argument('--latent_type', type=str, default='Ornstein-Uhlenbeck', help='type of GAN noise process')
    parser.add_argument('--latent_dim', type=int, default=64, help='dimension of GAN latent space')
    parser.add_argument('--latent_std_dev', type=float, default=0.3, help='standard deviation for GAN noise process')
    parser.add_argument('--time_emb_dim', type=int, default=76, help='dimension of the time embedding space')
    parser.add_argument('--input_units', type=int, default=16, help='number of neurons for feature extraction from input')
    parser.add_argument('--hidden_units', type=int, default=32, help='number of neurons for hidden layers')
    parser.add_argument('--num_hidden', type=int, default=4, help='number of hidden layers')
    parser.add_argument('--critic_steps_per_actor', type=int, default=5, help='critic updates for each actor update')
    parser.add_argument('--lambda_l1', type=float, default=10., help='coefficient for L1 losses')
    parser.add_argument('--lambda_grad_pen', type=float, default=10., help='coefficient for gradient penalization of critic')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='learning rate for actor network')
    parser.add_argument('--lr_inv_actor', type=float, default=1e-4, help='learning rate for inverse actor network')
    parser.add_argument('--lr_critic', type=float, default=1e-5, help='learning rate for critic network')
    parser.add_argument('--lr_inv_critic', type=float, default=1e-5, help='learning rate for inverse critic network')
    parser.add_argument('--lr_l1', type=float, default=1e-3, help='learning rate for L1 losses')
    parser.add_argument('--add_reward', type=bool, default=True, help='should we add the reward as input to critic network?')
    parser.add_argument('--add_time', type=bool, default=False, help='should we add time as input to all networks?')
    parser.add_argument('--var_horizon', type=bool, default=False, help='environment with a variable time horizon?')
    parser.add_argument('--tracks', type=str, default='fixed-start', help='family of tracks serving as demonstrations')
    parser.add_argument('--num_tracks', type=int, default=500, help='number of tracks in the family used for training')
    parser.add_argument('--save_dir', type=str, default='log', help='family of tracks serving as demonstrations')
    parser.add_argument('--train_runs', type=int, default=10, help='number of seeds for training')
    parser.add_argument('--test_runs', type=int, default=10, help='number of test runs per training seed')
    parser.add_argument('--render', type=bool, default=True, help='render the GYM environment during training/testing?')
    args = parser.parse_args()

    dati_builtin_tracks = tracks_env.built_in_tracks(args.timesteps, args.num_tracks)
    train_seeds = tracks_env.gen_seeds(args.train_runs)

    for seed_test, seed_train in enumerate(train_seeds):
        env = tracks_env.MouseHiddenCheese(
            tracks = dati_builtin_tracks[args.tracks], 
            num_episodes = args.num_episodes, 
            timesteps = args.timesteps, 
            need_unscaling=True, 
            seed=int(seed_train), 
            var_horizon=args.var_horizon,
        )
        model = DATI(env, render=args.render, save_dir=args.save_dir)
        logger = model.fit(**vars(args))
        model.evaluate(
            test_runs=args.test_runs, 
            seed_train=seed_train,
            seed_test=seed_test,
            last_episode=logger.episode
        )