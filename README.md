# Transferable Dynamics Learning

Benchmarking of different dynamics learning methods under changes in the distribution over controls. In this repository we provide the evaluation code and links to the corresponding evaluation data sets.

## Robotic setup

The data was recorded on a 3-DOF torque-controlled real-robotic system. Please check the [Open Dynamic Robot Initiative](https://open-dynamic-robot-initiative.github.io/) for the details about the robotic platform.

<img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/16.png" width="418"/> <img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/5.png" width="418"/>

## Robotic Dataset

All the dataset files can be downloaded from [here](https://owncloud.tuebingen.mpg.de/index.php/s/3THSfyBgFrYykPc?path=%2F)

We use the aforementioned robotic platform to record different datasets under substantially different conditions in order to assess the predictive performance of dynamics learning algorithms beyond the iid. setting. In this study such conditions are in turn governed by the particular controllers used to generate the control inputs feed into the system. We consider two types of controllers according to whether or not there is a feedback control loop.

### Closed-loop dataset - Sine waves PD control

This dataset was generated using a superposition of sine waves to generate trajectories that were subsequently tracked by the robot using PD position control (Please refer to the paper for details). Some sampled robot movements under the 4 different families of controllers considered in the closed-loop dataset:

<img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/1.gif" width="435"/><img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/3.gif" width="435"/>
<img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/2.gif" width="435"/><img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/4.gif" width="435"/>

The shown movements are arranged according to the following diagram where each of the datasets *D* account for a particular configuration of sine angular frenquencies (low, high) and reachable task space (left, full). The arrows denote different transfer settings, which are discussed in our paper.

<p align="center">
  <img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/datasets_closed_loop.png"/>
</p>

### Open-loop dataset - GP torque controls

We used sampled trajectories from a Gaussian process (GP) directly as torque inputs (controls) to record this dataset in an open loop fashion. Please refer to the paper for details.

### Dataset structure and naming convention.

The released files of the open-loop dataset are named according to the following convention:
* `Sines_full.npz`
* `Sines_training.npz`
* `Sines_validation.npz`
* `Sines_test_iid.npz`
* `Sines_test_transfer_%d.npz`

with the different test transfer files being indexed starting from 1 (e.g., `Sines_test_transfer_1.npz`). Note that each `_full.npz` files includes all recorded rollouts of its corresponding category. The rest of the files can be obtained from it as follows:

```
python -m DL.utils.data_extractor --raw_data Sines_full.npz --training Sines_training.npz --testiid Sines_test_iid.npz --validation Sines_validation.npz --testtransfer Sines_test_transfer_{}.npz
```

The open-loop dataset files are equally named, except that the prefix `GPs` is used instead of `Sines`.

Each of the released `npz` files can be indexed by the following keywords which account for different recorded variables:
* `measured_angles`
* `measured_velocities`
* `measured_torques`
* `constrained_torques`
* `desired_torques`

Each of these keys is associated with a numpy array of shape `(S, T, D)`, where `S` denotes the number of sequences/rollouts, and `T`, `D` are the sequence length and number of degrees of freedom, respectively. We recorded at a frequency of 1 Hz, meaning that the time elapsed between consecutive observations is 0.001 seconds. We recorded using a 3-DOF finger robot (`D=3`), and all rollouts have a total duration of 14s (`T=14000`).

## Simulated Dataset

TODO

## Reference

In order to cite us please do so according to:
```
@conference{AgudeloEspanaetal20,
  title = {A Real-Robot Dataset for Assessing Transferability of Learned Dynamics Models },
  author = {Agudelo-España, D. and Zadaianchuk, A. and Wenk, P. and Garg, A. and Akpo, J. and Grimminger, F. and Viereck, J. and Naveau, M. and Righetti, L. and Martius, G. and Krause, A. and Sch{\"o}lkopf, B. and Bauer, S. and W{\"u}thrich, M.},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2020}
}
```
