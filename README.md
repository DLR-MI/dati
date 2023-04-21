# DATI
This repository is the official implementation of the paper [Learning Representative Trajectories of Dynamical Systems via Domain-Adaptive Imitation](https://arxiv.org/abs/2304.10260).

<img src='assets/tracks.png'> 

DATI is a reinforcement learning agent for Domain-Adaptive Trajectory Imitation. It is trained from tracking data to learn how to generate the relevant spatio-temporal features using a cycle-consistent GAN approach. The figure above shows a toy example using a family of trajectories (built using the [OpenAI gym](https://github.com/openai/gym) framework) that serve as demonstrations for the imitation task (left); and typical rollouts from agents such as DDPG for trajectory imitation and Behavioral Cloning (right).

## Requirements
First clone this repository:
```
git clone https://github.com/DLR-MI/dati.git
cd dati
```
and install suitable dependencies:
- For pip users, you need to `pip install -r requirements.txt`.
- For conda users `conda env create -f environment.yml` and `conda activate dati`.

## Datasets
The episodes from the synthetic dataset generating the toy examples above can be generated using
```
python tracks_env.py
```
with the optional arguments

```commandline
--tracks TRACKS            TRACKS: defines the family of trajectories. One of [circles, ribbons, ushaped, fixed-start]
--n_tracks NTRACKS         NTRACKS: maximum number of trajectories to draw from family
--timesteps TIMESTEPS      TIMESTEPS: number of timesteps defining an episode (whole trajectory).
```
For a real dataset containing tracking data, the respective trajectories must be sampled using the same `track_envs.Tracks` class API. In the paper, we do this with vessel self-reporting position information from [MarineCadastre](https://marinecadastre.gov/AIS/).

## Train / Test
To train DATI and evaluate it on the toy examples, just type:
```
python run.py --num_episodes 100 --timesteps 200 --tracks circles
```
For a list of all supported arguments:
```
python run.py --help
```
After training and evaluating DATI, results are placed in `./log`. In order to generate those of the paper (assuming that you have generated similar log folders for `DDPG-TI`, `BC` and `ablation`):
```
python gen_results.py --dir_dati log_dati --dir_ddpg_ti log_ddpg_ti --dir_il log_bc
```

## Results
DATI outperforms popular baselines in all the tasks of the toy examples. Most interestingly, when applied to a real-world scenario, it can be used to detect abnormal trajectories (with a weighted F1-score of 0.78 in the case studied).

<img src='assets/abnormal.png'> 

## Citation

```
@misc{solanocarrillo2023learning,
      title={Learning Representative Trajectories of Dynamical Systems via Domain-Adaptive Imitation}, 
      author={Edgardo Solano-Carrillo and Jannis Stoppe},
      year={2023},
      eprint={2304.10260},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```