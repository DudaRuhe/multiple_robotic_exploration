# Mobile Multiple Robotic Exploration using PPO

This repository contains the code to train one or more agents to explore unknown environemnts using the [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) algorithm, without generetating a map.
This is a continuation of Luiza Garaffa's masters work.

**State Space**: Laser measurements + Agents' trajectories

**Action Space**: Up, Down, Left, Rigth.

## How to use it
1) Install the dependencies
```
make sure your python version is 3.5 or up
```
```
pip install gym
```
```
pip install pygame
```
```
pip install torch
```
```
pip install tensorboard
```
```
pip install setuptools==59.5.0
```

2) To perform trains or inferences with a single agent, move to the `single_agent` folder
```
cd single_agent
```
3) To perform trains or inferences with two or more agents, move to the `multiple_agents` folder 
```
cd multiple_agents
```
4) Configure the parameters in the `parameters.json` file
5) To train a new model, run:
```
python train.py
```
6) To perform inference with a single agent with an already trained model, make sure the model is saved in the trained_models folder, than run:
```
python infer.py
```

## Examples
### Single agent exploration in different maps (inference using training models)

<p align="center">
    <img src="https://user-images.githubusercontent.com/51202713/213886544-814151be-3457-4948-801c-74fe51fce85e.gif" width="40%" height="40%" 
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/51202713/213886764-8499eb8e-88e8-4e69-8ef9-b0f4305587b3.gif" width="40%" height="40%" 
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/51202713/213886963-dfa54723-9a31-481f-b62a-452cd2907ae5.gif" width="40%" height="40%" 
</p>

### Two agents exploration in different maps

<p align="center">
    <img src="https://user-images.githubusercontent.com/51202713/213886824-65f6a857-7ca0-4170-b528-2e16a1f5f922.gif" width="40%" height="40%" 
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/51202713/213886883-b6c54964-9e37-4017-b495-18e1ec3b89bb.gif" width="40%" height="40%" 
</p>!

<p align="center">
    <img src="https://user-images.githubusercontent.com/51202713/213887068-73803637-b2b3-49ac-80a9-f4a16df252be.gif" width="40%" height="40%" 
</p>!



