import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tracks_env import *

results = {
    'DATI': {
        'fixed-start': {
            'dtw_min': 0,
            'rollout': []
        },
        'u-shaped': {
            'dtw_min': 0,
            'rollout': []
        },  
        'circles': {
            'dtw_min': 0,
            'rollout': []
        },
        'ribbons': {
            'dtw_min': 0,
            'rollout': []
        },
    },
    'DDPG-TI': {
        'fixed-start': {
            'dtw_min': 0,
            'rollout': []
        },
        'u-shaped': {
            'dtw_min': 0,
            'rollout': []
        },  
        'circles': {
            'dtw_min': 0,
            'rollout': []
        },
        'ribbons': {
            'dtw_min': 0,
            'rollout': []
        },
    },
    'BC': {
        'fixed-start': {
            'dtw_min': 0,
            'rollout': []
        },
        'u-shaped': {
            'dtw_min': 0,
            'rollout': []
        },  
        'circles': {
            'dtw_min': 0,
            'rollout': []
        },
        'ribbons': {
            'dtw_min': 0,
            'rollout': []
        },
    },
}

def plot_families(save_dir=None, timesteps=200, ntracks=500, alpha=0.005):

    fig, axs = plt.subplots(1,2,figsize=(6.4,3.4))
    FixedStart(timesteps=timesteps, n_tracks=ntracks, seed=0
                ).plot_family(ax=axs[0], alpha=alpha)
    # axs[0].set_title('FixedStart', fontsize=13)
    UShaped(timesteps=timesteps, n_tracks=ntracks, seed=0
                    ).plot_family(ax=axs[1], alpha=alpha)
    # axs[1].set_title('FreeStart', fontsize=13)
    fig.tight_layout()
    if save_dir: 
        plt.savefig(f'{save_dir}/family_kw.svg  ')
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(6.4,3.4))
    Circles(timesteps=timesteps, n_tracks=ntracks, seed=0
            ).plot_family(ax=axs[0], alpha=alpha)
    # axs[0].set_title('Circles', fontsize=13)
    axs[0].set_aspect('equal', 'box')
    Ribbons(timesteps=timesteps, n_tracks=ntracks, seed=0
                ).plot_family(ax=axs[1], vars=[0, np.pi/2, np.pi], alpha=alpha)
    # axs[1].set_title('Ribbons', fontsize=13)
    axs[1].set_aspect('equal', 'box') 
    fig.tight_layout() 
    if save_dir:
        plt.savefig(f'{save_dir}/family_cr.svg')
    plt.show()

def get_best_from_methods(dir_dati=None, dir_ddpg_ti=None, dir_il=None):
    method = ['DATI', 'DDPG-TI', 'BC']
    tracks_names = ['fixed-start', 'u-shaped', 'circles', 'ribbons']
    
    for name in tracks_names:
        r = pd.read_csv(f'{dir_dati}/test_dati_{name}.csv')
        rg = pd.read_csv(f'{dir_dati}_gauss/test_dati_{name}.csv')
        seed, dtw_min = r[r.bestDTW == r.bestDTW.min()].values[0]
        _, dtw_min_g = rg[rg.bestDTW == rg.bestDTW.min()].values[0]
        dtw_min = dtw_min_g if name == 'ribbons' else dtw_min
        suffix = '_gauss' if name == 'ribbons' else ''
        with open(f'{dir_dati}{suffix}/rollouts_{name}_{int(seed)}.pkl', 'rb') as f:
            _, all_xy_actions = pickle.load(f)

        results['DATI'][name]['dtw_min'] = dtw_min
        results['DATI'][name]['rollout'] = all_xy_actions[0]

        r = pd.read_csv(f'{dir_ddpg_ti}/test_dati_{name}.csv')
        seed, dtw_min = r[r.bestDTW == r.bestDTW.min()].values[0]
        with open(f'{dir_ddpg_ti}/rollouts_{name}_{int(seed)}.pkl', 'rb') as f:
            _, all_xy_actions = pickle.load(f)

        results['DDPG-TI'][name]['dtw_min'] = dtw_min
        results['DDPG-TI'][name]['rollout'] = all_xy_actions[0][:-1,:]

        for k in range(2, len(method)):
            r = pd.read_csv(f'{dir_il}/{method[k]}/dtw_distances_{name}.csv')
            seed, dtw_min = r[r.dtw_distance == r.dtw_distance.min()].values[0]
            with open(f'{dir_il}/{method[k]}/rollouts_{name}.pkl', 'rb') as f:
                _, all_xy_actions = pickle.load(f)
            results[method[k]][name]['dtw_min'] = dtw_min
            results[method[k]][name]['rollout'] = all_xy_actions[int(seed)]
    return results, tracks_names

def display_results(**kwargs):
    timesteps, ntracks = kwargs['timesteps'], kwargs['ntracks']
    r, names = get_best_from_methods(kwargs['dir_dati'], kwargs['dir_ddpg_ti'], kwargs['dir_il'])
    df = {}
    for name in names:
        df[name] = [
            r['DATI'][name]['dtw_min'],
            r['DDPG-TI'][name]['dtw_min'],
            r['BC'][name]['dtw_min'],
        ]
    df = pd.DataFrame(df, index=['DATI', 'DDPG-TI', 'BC'])
    pd.options.display.float_format = '{:,.3f}'.format
    print(df)

    fig, axs = plt.subplots(1,2,figsize=(6.4,3.4))

    axs[0].plot(
        r['DATI']['fixed-start']['rollout'][:,0],
        r['DATI']['fixed-start']['rollout'][:,1],
        linewidth=2,
        label = 'DATI'
    )
    axs[0].plot(
        r['BC']['fixed-start']['rollout'][:,0],
        r['BC']['fixed-start']['rollout'][:,1],
        label = 'BC',
        alpha = kwargs['alpha_line'],
        linestyle = '--'
    )
    axs[0].plot(
        r['DDPG-TI']['fixed-start']['rollout'][:,0],
        r['DDPG-TI']['fixed-start']['rollout'][:,1],
        label = 'DDPG-TI',
        alpha = kwargs['alpha_line'],
        linestyle = ':'
    )
    FixedStart(timesteps=timesteps, n_tracks=ntracks, seed=0
                ).plot_family(ax=axs[0], alpha=kwargs['alpha_shadow'], select=False)
    # axs[0].set_title('FixedStart', fontsize=13)
    axs[0].legend(fontsize=13.5)

    axs[1].plot(
        r['DATI']['u-shaped']['rollout'][:,0],
        r['DATI']['u-shaped']['rollout'][:,1],
        linewidth=2,
    )
    axs[1].plot(
        r['BC']['u-shaped']['rollout'][:,0],
        r['BC']['u-shaped']['rollout'][:,1],
        alpha = kwargs['alpha_line'],
        linestyle = '--'
    )
    axs[1].plot(
        r['DDPG-TI']['u-shaped']['rollout'][:,0],
        r['DDPG-TI']['u-shaped']['rollout'][:,1],
        alpha = kwargs['alpha_line'],
        linestyle = ':'
    )
    UShaped(timesteps=timesteps, n_tracks=ntracks, seed=0
                    ).plot_family(ax=axs[1], alpha=kwargs['alpha_shadow'], select=False)
    # axs[1].set_title('FreeStart', fontsize=13)

    fig.tight_layout()
    if kwargs['save_dir']: 
        plt.savefig(f"{kwargs['save_dir']}/results_kw.svg")
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(6.4,3.4))

    axs[0].plot(
        r['DATI']['circles']['rollout'][:,0],
        r['DATI']['circles']['rollout'][:,1],
        linewidth=2,
    )
    axs[0].plot(
        r['BC']['circles']['rollout'][:,0],
        r['BC']['circles']['rollout'][:,1],
        alpha = kwargs['alpha_line'],
        linestyle = '--'
    )
    axs[0].plot(
        r['DDPG-TI']['circles']['rollout'][:,0],
        r['DDPG-TI']['circles']['rollout'][:,1],
        alpha = kwargs['alpha_line'],
        linestyle = ':'
    )
    Circles(timesteps=timesteps, n_tracks=ntracks, seed=0
            ).plot_family(ax=axs[0], alpha=kwargs['alpha_shadow'], select=False)
    # axs[0].set_title('Circles', fontsize=13)
    axs[0].set_aspect('equal', 'box')

    axs[1].plot(
        r['DATI']['ribbons']['rollout'][:,0],
        r['DATI']['ribbons']['rollout'][:,1],
        linewidth=2,
    )
    axs[1].plot(
        r['BC']['ribbons']['rollout'][:,0],
        r['BC']['ribbons']['rollout'][:,1],
        alpha = kwargs['alpha_line'],
        linestyle = '--'
    )
    axs[1].plot(
        r['DDPG-TI']['ribbons']['rollout'][:,0],
        r['DDPG-TI']['ribbons']['rollout'][:,1],
        alpha = kwargs['alpha_line'],
        linestyle = ':'
    )
    Ribbons(timesteps=timesteps, n_tracks=ntracks, seed=0
                ).plot_family(ax=axs[1], vars=[0, np.pi/2, np.pi], 
                              alpha=kwargs['alpha_shadow'], select=False)
    # axs[1].set_title('Ribbons', fontsize=13)
    axs[1].set_aspect('equal', 'box') 
    fig.tight_layout() 
    if kwargs['save_dir']:
        plt.savefig(f"{kwargs['save_dir']}/results_cr.svg")
    plt.show()

def ablation_results(dir_original=None, dir_ablation=None):
    pd.options.display.float_format = '{:,.3f}'.format
    r = pd.read_csv(f'{dir_original}/test_dati_circles.csv')
    r_nt = pd.read_csv(f'{dir_ablation}/no_time/test_dati_circles.csv')
    r_nr = pd.read_csv(f'{dir_ablation}/no_reward/test_dati_circles.csv')
    r_tg = pd.read_csv(f'{dir_ablation}/noise_gauss/test_dati_circles.csv')
    dtw_min, dtw_min2 = r.bestDTW.nsmallest(2)
    dtw_min_nt, dtw_min_nt2 = r_nt.bestDTW.nsmallest(2)
    dtw_min_nr, dtw_min_nr2 = r_nr.bestDTW.nsmallest(2)
    dtw_min_tg, dtw_min_tg2 = r_tg.bestDTW.nsmallest(2)
    print(
        pd.DataFrame(
            {
                'Transformation': [
                    'Original setup',
                    'No time embedding',
                    'No reward reinforcement',
                    'Changing noise process'
                ],
                'dtw_min': [
                    dtw_min,
                    dtw_min_nt,
                    dtw_min_nr,
                    dtw_min_tg
                ],
                'dtw_min2': [
                    dtw_min2,
                    dtw_min_nt2,
                    dtw_min_nr2,
                    dtw_min_tg2
                ]
            }
        )
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_dati', type=str, required=True, help='path to log files from dati model')
    parser.add_argument('--dir_ddpg_ti', type=str, required=True, help='path to log files from ddpg_ti model')
    parser.add_argument('--dir_il', type=str, required=True, help='path to log files from bc model')
    parser.add_argument('--dir_ablation', type=str, required=True, help='path to log files from ablation studies')
    parser.add_argument('--dir_save', default=None, help='path to save figures')
    args = parser.parse_args()

    plot_families(save_dir = args.dir_save)

    display_results(
        save_dir = args.dir_save,
        timesteps = 100,
        ntracks = 500,
        alpha_shadow = 0.005,
        alpha_line = 0.9,
        dir_dati = args.dir_dati,
        dir_ddpg_ti = args.dir_ddpg_ti,
        dir_il = args.dir_il
    )

    ablation_results(
        dir_original = args.dir_dati,
        dir_ablation = args.dir_ablation
    )