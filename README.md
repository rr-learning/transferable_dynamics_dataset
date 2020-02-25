# Transferable Dynamics Learning

Benchmarking of different dynamics learning methods under changes in the distribution over controls. In this repository we provide the evaluation code and links to the correspoding evaluation data sets.

## Robotic setup

The data was recorded on a 3-DOF torque-controlled real-robotic system. Please check the [Open Dynamic Robot Initiative](https://open-dynamic-robot-initiative.github.io/) for the details about the robotic platform.

<img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/16.png" width="418"/> <img src="https://github.com/rr-learning/transferable_dynamics_dataset/blob/master/img/5.png" width="418"/>

## Robotic Dataset

We use the aforementioned robotic platform to record different datasets under substantially different conditions in order to assess the predictive performance of dynamics learning algorithms beyond the iid. setting. Each of the released datasets is a `.npz` file which can be indexed by the following keywords accounting for different recorded variables:

* `measured_angles`
* `measured_velocities`
* `measured_torques`
* `constrained_torques`
* `desired_torques`

Each of these keys is associated with a numpy array of shape `(S, T, D)`, where `S` denotes the number of sequences/rollouts, and `T`, `D` are the sequence length and number of degrees of freedom, respectively. We recorded at a frequency of 1 Hz, meaning that the time elapsed between consecutive observations is 0.001 seconds. We recorded using a 3-DOF finger robot (`D=3`), and all rollouts have a total duration of 14s (`T=14000`).

### Closed-loop dataset - Sine waves PD control

This dataset was generated using a superposition of sine waves to generate trajectories that were subsequently tracked by the robot using PD position control. Please refer to the paper for details.

### Open-loop dataset - GP torque controls

We used sampled trajectories from a GP directly as torque inputs (controls) to record this dataset in an open loop fashion. Please refer to the paper for details.

## Simulated Dataset

## Reference

TODO
