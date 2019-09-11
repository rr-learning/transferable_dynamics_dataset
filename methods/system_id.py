from __future__ import print_function

import argparse
import sys


from collections import OrderedDict 


import numpy as np
import numpy.matlib
import time
import os

import math

import matplotlib.pylab as plt

import pinocchio

from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import *
from os.path import join, dirname

from scipy.ndimage import gaussian_filter1d

import rospkg

import time

import cvxpy

# dynamics learning stuff
from DL import DynamicsLearnerInterface


class SystemId(DynamicsLearnerInterface):
    def __init__(self,
                 history_length,
                 prediction_horizon,
                 settings=None):
        DynamicsLearnerInterface.__init__(self,
                                          history_length=history_length,
                                          prediction_horizon=prediction_horizon,
                                          difference_learning=False,
                                          averaging=False,
                                          streaming=False)
        if settings is None:
            settings = {}
        # TODO: should we have a default value for this attribute.
        self.identification_method = settings.pop(
            'identification_method', None)
        self.robot = Robot(**settings)

        self.dt = 0.001

    def learn(self, observation_sequences, action_sequences):
        # preprocess data ------------------------------------------------------
        data = dict()
        data['angle'] = observation_sequences[:, :, :3]
        data['velocity'] = observation_sequences[:, :, 3:6]
        data['torque'] = action_sequences
        compute_accelerations(data, self.dt)
        data = preprocess_data(data=data,
                               desired_n_data_points=100000,
                               smoothing_sigma=1.0)
        print('Learning with {} points'.format(data['angle'].shape[0]))

        # identify -------------------------------------------------------------
        if self.identification_method == 'cad':
            return
        elif self.identification_method == 'ls':
            sys_id(robot=self.robot,
                   angle=data['angle'],
                   velocity=data['velocity'],
                   acceleration=data['acceleration'],
                   torque=data['torque'])
        elif self.identification_method == 'ls-lmi':
            sys_id_lmi(robot=self.robot,
                       angle=data['angle'],
                       velocity=data['velocity'],
                       acceleration=data['acceleration'],
                       torque=data['torque'])
        else:
            raise NotImplementedError('Choose an identification method')

    def predict(self, observation_history, action_history, action_future=None):
        # parse arguments ------------------------------------------------------
        n_samples = observation_history.shape[0]
        dim_observation = observation_history.shape[2]

        if action_future is None:
            assert self.prediction_horizon == 1
            action_future = np.empty((n_samples, 0, self.action_dimension))

        assert (action_future.shape[1] == self.prediction_horizon - 1)

        # make predictions -----------------------------------------------------
        predictions = np.empty((n_samples, dim_observation))
        predictions[:] = numpy.nan

        for i in range(n_samples):
            torques_sequence = np.append(action_history[i, -1:],
                                         action_future[i],
                                         axis=0)

            angle = observation_history[i, -1, :3]
            velocity = observation_history[i, -1, 3:6]
            for t in range(0, self.prediction_horizon):
                angle, velocity = \
                    self.robot.predict(angle=angle,
                                       velocity=velocity,
                                       torque=torques_sequence[t],
                                       dt=self.dt)

            predictions[i] = np.concatenate([np.array(angle).flatten(),
                                             np.array(velocity).flatten(),
                                             torques_sequence[-1]], axis=0)
        return predictions

    def name(self):
        return 'system_id'


def to_matrix(array):
    matrix = np.matrix(array)
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.transpose()

    return matrix


def to_diagonal_matrix(vector):
    return np.matrix(np.diag(np.array(vector).flatten()))


class Robot(RobotWrapper):
    def __init__(self, symplectic=True, init='cad',
                 visualizer=None):
        self.load_urdf()
        self.viscous_friction = to_matrix(np.zeros(3)) + 0.01
        self.static_friction = to_matrix(np.zeros(3)) + 0.00
        self.symplectic = symplectic
        if visualizer == "meshcat":
            self.setVisualizer(MeshcatVisualizer())
        elif visualizer == "gepetto":
            self.setVisualizer(GepettoVisualizer())
        elif visualizer:
            raise NotImplementedError

        # Initialization of Parameters
        if init == 'cad':
            pass
        elif init == 'random':
            self.set_random_params()
        elif init == 'noisy':
            self.set_noisy_params()
        elif init == 'identity':
            self.set_identity_params()
        else:
            raise NotImplementedError

    # dynamics -----------------------------------------------------------------
    def simulate(self,
                 dt,
                 n_steps=None,
                 torque=None,
                 initial_angle=None,
                 initial_velocity=None,
                 mask=np.ones(3),
                 verbose=False):
        """ Returns the sequence of angles, velocities and torques resulting
            from simulating the given torques."""
        zero = pinocchio.utils.zero(self.model.nv)

        torque = np.array(zero) if torque is None else np.array(torque)
        torque = torque.reshape(-1, 3, 1)
        if torque.shape[0] == 1:
            assert (n_steps)
            torque = np.repeat(torque, repeats=n_steps, axis=0)
        elif n_steps:
            assert (n_steps == torque.shape[0])

        angle = zero if initial_angle is None else to_matrix(initial_angle)
        velocity = \
            zero if initial_velocity is None else to_matrix(initial_velocity)
        mask = to_matrix(mask)

        last_time = time.time()
        simulated_angles = []
        simulated_vels = []
        simulated_accelerations = []
        applied_torques = []
        for t in range(torque.shape[0]):
            acceleration = self.forward_dynamics(angle, velocity, torque[t])

            simulated_angles.append(np.ravel(angle))
            simulated_vels.append(np.ravel(velocity))
            simulated_accelerations.append(np.ravel(acceleration))
            applied_torques.append(np.ravel(torque[t]))
            if self.symplectic:
                velocity = velocity + np.multiply(mask, acceleration * dt)
                angle = angle + np.multiply(mask, velocity * dt)
            else:
                angle = angle + np.multiply(mask, velocity * dt)
                velocity = velocity + np.multiply(mask, acceleration * dt)
            if verbose:
                print('angle: ', np.array(angle).flatten(),
                      '\nvelocity: ', np.array(velocity).flatten())
        return np.array(simulated_angles), np.array(simulated_vels), np.array(simulated_accelerations), np.array(applied_torques)

    # TODO: this needs to be checked
    def predict(self, angle, velocity, torque, dt):
        angle = to_matrix(angle)
        velocity = to_matrix(velocity)
        torque = to_matrix(torque)
        acceleration = self.forward_dynamics(angle, velocity, torque)
        if self.symplectic:
            velocity = velocity + acceleration * dt
            angle = angle + velocity * dt
        else:
            angle = angle + velocity * dt
            velocity = velocity + acceleration * dt
        return angle, velocity

    def friction_torque(self, velocity):
        velocity = to_matrix(velocity)
        return -(np.multiply(velocity, self.viscous_friction) +
                 np.multiply(np.sign(velocity), self.static_friction))

    def forward_dynamics(self, angle, velocity, actuator_torque):
        joint_torque = actuator_torque + self.friction_torque(velocity)

        return pinocchio.aba(self.model, self.data, angle, velocity,
                             joint_torque)

    def inverse_dynamics(self, angle, velocity, acceleration):

        joint_torque = pinocchio.rnea(self.model, self.data,
                                      to_matrix(angle),
                                      to_matrix(velocity),
                                      to_matrix(acceleration))
        actuator_torque = joint_torque - self.friction_torque(velocity)

        # just as a sanity check -----------------------------------------------
        Y = self.compute_regressor_matrix(angle, velocity, acceleration)
        actuator_torque_1 = Y * self.get_params()
        assert ((abs(actuator_torque - actuator_torque_1) <= 1e-9).all())
        # ----------------------------------------------------------------------

        return actuator_torque

    def compute_regressor_matrix(self, angle, velocity, acceleration):
        joint_torque_regressor = \
            pinocchio.computeJointTorqueRegressor(self.model, self.data,
                                                  to_matrix(angle),
                                                  to_matrix(velocity),
                                                  to_matrix(acceleration))

        viscous_friction_torque_regressor = to_diagonal_matrix(velocity)
        static_friction_torque_regressor = to_diagonal_matrix(
            np.sign(velocity))

        regressor_matrix = np.concatenate([
            joint_torque_regressor,
            viscous_friction_torque_regressor,
            static_friction_torque_regressor], axis=1)

        return regressor_matrix

    # getters and setters ------------------------------------------------------
    def get_params(self):
        theta = [self.model.inertias[i].toDynamicParameters()
                 for i in range(1, len(self.model.inertias))]

        theta = theta + [self.viscous_friction, self.static_friction]

        theta = np.concatenate(theta, axis=0)
        return theta

    def get_com(self, link_index):
        raise NotImplementedError

    def get_mass(self, link_index):
        raise NotImplementedError

    def get_inertia_matrix_link_frame(self, link_index):
        return np.array(self.model.inertias[link_index + 1].inertia)

    def set_params(self, theta):

        for dof in range(self.model.nv):
            theta_dof = theta[dof * 10: (dof + 1) * 10]

            self.model.inertias[dof + 1] = \
                pinocchio.libpinocchio_pywrap.Inertia.FromDynamicParameters(
                    theta_dof)

        n_inertial_params = self.model.nv * 10
        self.viscous_friction = theta[n_inertial_params: n_inertial_params + 3]
        self.static_friction = theta[
            n_inertial_params + 3: n_inertial_params + 6]

        assert (((self.get_params() - theta) < 1e-9).all())

    def set_random_params(self):
        for dof in range(self.model.nv):
            self.model.inertias[dof + 1].setRandom()

    def set_identity_params(self):
        for dof in range(self.model.nv):
            self.model.inertias[dof + 1].setIdentity()

    def set_noisy_params(self):
        sigma = 0.001
        for dof in range(self.model.nv):
            self.model.inertias[dof + 1].mass += sigma * np.random.randn()
            self.model.inertias[dof + 1].lever += sigma * np.random.randn(3, 1)
            self.model.inertias[dof + 1].inertia += np.abs(np.diag(
                sigma * np.random.randn(3)))

    # loading ------------------------------------------------------------------
    def load_urdf(self):
        try:
            model_path = rospkg.RosPack().get_path(
                "robot_properties_manipulator")
        except rospkg.ResourceNotFound:
            print('Warning: The URDF is not being loaded from a ROS package.')
            current_path = str(os.path.dirname(os.path.abspath(__file__)))
            model_path = str(os.path.abspath(os.path.join(current_path,
                                                          '../../robot_properties_manipulator')))
        urdf_path = join(model_path, "urdf", "manipulator.urdf")
        meshes_path = dirname(model_path)
        print(urdf_path, meshes_path)
        self.initFromURDF(urdf_path, [meshes_path])


def test_regressor_matrix(robot):
    for _ in range(100):
        angle = pinocchio.randomConfiguration(robot.model)
        velocity = pinocchio.utils.rand(robot.model.nv)
        acceleration = pinocchio.utils.rand(robot.model.nv)

        Y = robot.compute_regressor_matrix(angle, velocity, acceleration)
        theta = robot.get_params()
        other_tau = Y * theta

        torque = robot.inverse_dynamics(angle, velocity, acceleration)

        assert ((abs(torque - other_tau) <= 1e-9).all())


def load_data():
    all_data = np.load(args.input)
    data = dict()
    data['angle'] = all_data['measured_angles']
    data['velocity'] = all_data['measured_velocities']

    # TODO: not sure whether to take measured or constrained torques
    data['torque'] = all_data['measured_torques']
    return data


def show_angle_trajectory(q_trajectory, dt=0.001):
    robot = Robot(visualizer=args.visualizer)
    robot.initViewer(loadModel=True)
    robot.play(q_trajectory=q_trajectory, dt=0.001)


def compute_accelerations(data, dt):
    data['acceleration'] = np.diff(data['velocity'], axis=1) / dt
    for key in ['angle', 'velocity', 'torque']:
        data[key] = data[key][:, :-1]

    # test that everything worked out -----------------------------------------
    integrated_velocity = data['velocity'][:, :-1] + \
        data['acceleration'][:, :-1] * dt

    is_consistent = (np.absolute(integrated_velocity - data['velocity'][:, 1:]) <= 1e-12).all()
    assert(is_consistent)


def preprocess_data(data, desired_n_data_points,
                    smoothing_sigma=None, shuffle_data=True):
    # smoothen -----------------------------------------------------------------
    if smoothing_sigma is not None:
        for key in data.keys():
            data[key] = gaussian_filter1d(data[key],
                                          sigma=smoothing_sigma,
                                          axis=1)

    # cut off ends -------------------------------------------------------------
    for key in data.keys():
        data[key] = data[key][:,
                              data[key].shape[1] // 10: -data[key].shape[1] // 10]

    # reshape ------------------------------------------------------------------
    ordered_data = data.copy()
    n_trajectories = data['angle'].shape[0]
    n_time_steps = data['angle'].shape[1]
    n_data_points = n_trajectories * n_time_steps
    for key in data.keys():
        data[key] = np.reshape(data[key], [n_data_points, 3])

    for _ in range(10):
        trajectory_idx = np.random.randint(n_trajectories)
        time_step = np.random.randint(n_time_steps)

        global_idx = trajectory_idx * n_time_steps + time_step

        for key in data.keys():
            assert ((ordered_data[key][trajectory_idx, time_step] ==
                     data[key][global_idx]).all())

    # return a random subset of the datapoints ---------------------------------
    data_point_indices = np.arange(n_data_points)
    if shuffle_data:
        data_point_indices = np.random.permutation(data_point_indices)

    if desired_n_data_points < n_data_points:
        data_point_indices = data_point_indices[:desired_n_data_points]

    for key in data.keys():
        data[key] = data[key][data_point_indices]

    return data


def satisfies_normal_equation(theta, Y, T, epsilon=1e-6):
    lhs = (Y.transpose() * Y).dot(theta)
    rhs = Y.transpose().dot(T)
    return (abs(lhs - rhs) < epsilon).all()


def rmse_sequential(robot, angle, velocity, acceleration, torque):
    sum_squared_error = 0

    T = angle.shape[0]

    for t in range(T):
        predicted_torque = robot.inverse_dynamics(angle=angle[t],
                                                  velocity=velocity[t],
                                                  acceleration=acceleration[t])
        sum_squared_error = sum_squared_error \
            + np.linalg.norm(
                predicted_torque - to_matrix(torque[t])) ** 2

    mean_squared_error = sum_squared_error / T / 3

    return np.sqrt(mean_squared_error)


def rmse_batch(theta, Y, T):
    return np.squeeze(np.sqrt(
        (Y * theta - T).transpose().dot(Y * theta - T) / len(T)))


def sys_id_lmi(robot, angle, velocity, acceleration, torque):
    log = dict()

    test_regressor_matrix(robot)

    Y = np.concatenate(
        [robot.compute_regressor_matrix(angle[t], velocity[t], acceleration[t])
         for t in
         range(angle.shape[0])], axis=0)

    T = np.concatenate(
        [to_matrix(torque[t]) for t in range(angle.shape[0])], axis=0)

    log['rmse_sequential_before_id'] = rmse_sequential(robot=robot,
                                                       angle=angle,
                                                       velocity=velocity,
                                                       acceleration=acceleration,
                                                       torque=torque)
    log['rmse_batch_before_id'] = rmse_batch(theta=robot.get_params(),
                                             Y=Y, T=T)

    # Constrained optimization using CVXPY
    # theta = [m, mc_x, mc_y, mc_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]
    theta = cvxpy.Variable((36, 1))

    prior = robot.get_params().A
    theta = cvxpy.Variable((36, 1))
    cost = cvxpy.norm(Y * theta - T, 1) + 1e-6 * cvxpy.norm(
        theta - prior, 2)

    I = []
    J = []
    m = []
    mc = []
    for i in range(3):
        m_i = theta[10 * i]
        mc_i = [theta[10 * i + 1], theta[10 * i + 2], theta[10 * i + 3]]

        I_i = [[theta[10 * i + 4], theta[10 * i + 5], theta[10 * i + 7]],
               [theta[10 * i + 5], theta[10 * i + 6], theta[10 * i + 8]],
               [theta[10 * i + 7], theta[10 * i + 8], theta[10 * i + 9]]]

    fr_v = [theta[30], theta[31], theta[32]]
    fr_s = [theta[33], theta[34], theta[35]]

    J = []
    constraints = []
    for i in range(3):
        J += [cvxpy.bmat([
            [0.5 * (-theta[i * 10 + 4] + theta[i * 10 + 6] + theta[i * 10 + 9]),
             -theta[i * 10 + 5],
             -theta[i * 10 + 7],
             theta[i * 10 + 1]],
            [
                -theta[i * 10 + 5],
                0.5 * (theta[i * 10 + 4] - theta[i * 10 + 6] + theta[
                    i * 10 + 9]),
                -theta[i * 10 + 8],
                theta[i * 10 + 2]],
            [
                -theta[i * 10 + 7],
                -theta[i * 10 + 8],
                0.5 * (theta[i * 10 + 4] + theta[i * 10 + 6] - theta[
                    i * 10 + 9]),
                theta[i * 10 + 3]],
            [
                theta[i * 10 + 1],
                theta[i * 10 + 2],
                theta[i * 10 + 3],
                theta[i * 10]]])]

        constraints += [J[i] >> 0]
        constraints += [theta[10 * i + 4] <= 0.005]
        constraints += [theta[10 * i + 6] <= 0.005]
        constraints += [theta[10 * i + 9] <= 0.005]

    mass_indices = [0, 10, 20]
    for i in mass_indices:
        constraints += [theta[i] <= 0.3] + [theta[i] >= 0.05]

    friction_indices = [30, 31, 32, 33, 34, 35]
    for i in friction_indices:
        constraints += [theta[i] >= 0]

    com_indices = [1, 2, 3, 11, 12, 13, 21, 22, 23]
    for i in com_indices:
        constraints += [theta[com_indices] <= 0.03] + [
            theta[com_indices] >= -0.03]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(verbose=True, eps=10e-6, max_iters=2000, solver='SCS')

    theta = theta.value

    log['rmse_batch_optimal_theta_after_id_lmi'] = rmse_batch(theta=theta,
                                                              Y=Y, T=T)

    robot.set_params(theta)

    test_regressor_matrix(robot)

    log['rmse_sequential_after_id_lmi'] = rmse_sequential(robot=robot,
                                                          angle=angle,
                                                          velocity=velocity,
                                                          acceleration=acceleration,
                                                          torque=torque)
    log['rmse_batch_after_id_lmi'] = rmse_batch(theta=robot.get_params(), Y=Y,
                                                T=T)

    for key in log.keys():
        print(key + ': ', log[key], '\n')

    for i in range(len(robot.model.inertias)):
        print(robot.model.inertias[i])

    print('static_friction: ', robot.static_friction, '\n',
          'visous_friction: ', robot.viscous_friction)


def sys_id_lmi_diagonal(robot, angle, velocity, acceleration, torque):
    log = dict()

    test_regressor_matrix(robot)

    Y = np.concatenate(
        [robot.compute_regressor_matrix(angle[t], velocity[t], acceleration[t])
         for t in
         range(angle.shape[0])], axis=0)

    T = np.concatenate(
        [to_matrix(torque[t]) for t in range(angle.shape[0])], axis=0)

    log['rmse_sequential_before_id'] = rmse_sequential(robot=robot,
                                                       angle=angle,
                                                       velocity=velocity,
                                                       acceleration=acceleration,
                                                       torque=torque)
    log['rmse_batch_before_id'] = rmse_batch(theta=robot.get_params(),
                                             Y=Y, T=T)

    # Constrained optimization using CVXPY
    # theta = [m, mc_x, mc_y, mc_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]
    theta = cvxpy.Variable((36, 1))

    prior = robot.get_params().A
    theta = cvxpy.Variable((36, 1))
    cost = cvxpy.sum_squares(Y * theta - T) + 1e-6 * cvxpy.sum_squares(
        theta - prior)

    J = []
    constraints = []
    for i in range(3):
        J += [cvxpy.bmat([
            [0.5 * (-theta[i * 10 + 4] + theta[i * 10 + 6] + theta[i * 10 + 9]),
             -theta[i * 10 + 5],
             -theta[i * 10 + 7],
             theta[i * 10 + 1]],
            [
                -theta[i * 10 + 5],
                0.5 * (theta[i * 10 + 4] - theta[i * 10 + 6] + theta[
                    i * 10 + 9]),
                -theta[i * 10 + 8],
                theta[i * 10 + 2]],
            [
                -theta[i * 10 + 7],
                -theta[i * 10 + 8],
                0.5 * (theta[i * 10 + 4] + theta[i * 10 + 6] - theta[
                    i * 10 + 9]),
                theta[i * 10 + 3]],
            [
                theta[i * 10 + 1],
                theta[i * 10 + 2],
                theta[i * 10 + 3],
                theta[i * 10]]])]

        constraints += [J[i] >> 0]

    mass_indices = [0, 10, 20]
    for i in mass_indices:
        constraints += [theta[i] <= 0.3] + [theta[i] >= 0.05]

    friction_indices = [30, 31, 32, 33, 34, 35]
    for i in friction_indices:
        constraints += [theta[i] >= 0]

    com_indices = [1, 2, 3, 11, 12, 13, 21, 22, 23]
    for i in com_indices:
        constraints += [theta[com_indices] <= 0.03] + [
            theta[com_indices] >= -0.03]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(verbose=False, eps=10e-6, max_iters=10000, solver='SCS')

    theta = theta.value

    log['rmse_batch_optimal_theta_after_id_lmi'] = rmse_batch(theta=theta,
                                                              Y=Y, T=T)

    robot.set_params(theta)

    test_regressor_matrix(robot)

    log['rmse_sequential_after_id_lmi'] = rmse_sequential(robot=robot,
                                                          angle=angle,
                                                          velocity=velocity,
                                                          acceleration=acceleration,
                                                          torque=torque)
    log['rmse_batch_after_id_lmi'] = rmse_batch(theta=robot.get_params(), Y=Y,
                                                T=T)

    for key in log.keys():
        print(key + ': ', log[key], '\n')

    for i in range(len(robot.model.inertias)):
        print(robot.model.inertias[i])

    print('static_friction: ', robot.static_friction, '\n',
          'visous_friction: ', robot.viscous_friction)

    # ipdb.set_trace()


def sys_id(robot, angle, velocity, acceleration, torque):
    log = dict()

    test_regressor_matrix(robot)

    Y = np.concatenate(
        [robot.compute_regressor_matrix(angle[t], velocity[t], acceleration[t])
         for t in
         range(angle.shape[0])], axis=0)

    T = np.concatenate(
        [to_matrix(torque[t]) for t in range(angle.shape[0])], axis=0)

    log['rmse_sequential_before_id'] = rmse_sequential(robot=robot,
                                                       angle=angle,
                                                       velocity=velocity,
                                                       acceleration=acceleration,
                                                       torque=torque)
    log['rmse_batch_before_id'] = rmse_batch(theta=robot.get_params(),
                                             Y=Y, T=T)

    regularization_epsilon = 1e-10
    regularization_mu = 1e-6
    theta = np.linalg.solve(
        Y.transpose() * Y + regularization_epsilon * np.eye(Y.shape[1],
                                                            Y.shape[1]),
        Y.transpose() * T + regularization_epsilon * regularization_mu)

    log['rmse_batch_optimal_theta_after_id'] = rmse_batch(theta=theta,
                                                          Y=Y, T=T)

    # for key in log.keys():
    #     print(key + ': ', log[key], '\n')

    # ipdb.set_trace()

    robot.set_params(theta)

    # TODO: include the regularization in the normal equation assertion.
    # assert (satisfies_normal_equation(robot.get_params(), Y, T))
    test_regressor_matrix(robot)

    log['rmse_sequential_after_id'] = rmse_sequential(robot=robot,
                                                      angle=angle,
                                                      velocity=velocity,
                                                      acceleration=acceleration,
                                                      torque=torque)
    log['rmse_batch_after_id'] = rmse_batch(theta=robot.get_params(), Y=Y, T=T)

    for key in log.keys():
        print(key + ': ', log[key], '\n')


def test_sys_id_simulated_torques():
    robot = Robot()
    test_regressor_matrix(robot)

    # create dataset with simulated torques ------------------------------------
    data = load_data()
    compute_accelerations(data, dt=0.001)

    for key in data.keys():
        data[key] = data[key][:, 10000: 10200]

    data['torque'] = [[robot.inverse_dynamics(
        angle=data['angle'][trajectory_idx, t],
        velocity=data['velocity'][trajectory_idx, t],
        acceleration=data['acceleration'][trajectory_idx, t])
        for t in range(data['angle'].shape[1])]
        for trajectory_idx in range(data['angle'].shape[0])]
    data['torque'] = np.array(data['torque']).squeeze()

    # the usual preprocessing --------------------------------------------------
    data = preprocess_data(data=data,
                           desired_n_data_points=10000,
                           smoothing_sigma=None)

    # identify -----------------------------------------------------------------
    sys_id(robot=robot,
           angle=data['angle'],
           velocity=data['velocity'],
           acceleration=data['acceleration'],
           torque=data['torque'])

    assert (rmse_sequential(robot=robot,
                            angle=data['angle'],
                            velocity=data['velocity'],
                            acceleration=data['acceleration'],
                            torque=data['torque']) < 1e-10)


def test_sys_id_visually():
    assert args.visualizer
    robot = Robot(visualizer=args.visualizer)
    robot.simulate(dt=0.001,
                   n_steps=1000,
                   torque=[0.1, 0.1, 0.1],
                   initial_angle=[1, 1, 1],
                   mask=[1, 1, 1])

    # robot.simulate(dt=0.001, n_steps=10000)

    data = load_data()
    compute_accelerations(data, dt=0.001)

    # # plot ---------------------------------------------------------------------
    # for key in data.keys():
    #     data[key] = gaussian_filter1d(data[key],
    #                                   sigma=3,
    #                                   axis=1)
    #
    #
    # dim = 2
    # sample = 50
    # data['velocity'] /= 20
    # stuff = ['torque', 'acceleration', 'velocity']
    # for key in stuff:
    #     plt.plot(data[key][sample, 1000:2000, dim])
    #
    # plt.legend(stuff)
    #
    # # plt.ylim([-10, 10])
    #
    # plt.show()
    # ipdb.set_trace()

    data = preprocess_data(data=data,
                           desired_n_data_points=10000,
                           smoothing_sigma=1)

    sys_id(robot=robot,
           angle=data['angle'],
           velocity=data['velocity'],
           acceleration=data['acceleration'],
           torque=data['torque'])

    # robot.simulate(dt=0.001, n_steps=10000)


def test_sys_id_lmi():
    robot = Robot()
    # robot.simulate(dt=0.001,
    #                n_steps=1000,
    #                torque=[0.1, 0.1, 0.1],
    #                initial_angle=[1, 1, 1],
    #                mask=[1, 1, 1])

    # robot.simulate(dt=0.001, n_steps=10000)

    data = load_data()
    compute_accelerations(data, dt=0.001)

    # # plot ---------------------------------------------------------------------
    # for key in data.keys():
    #     data[key] = gaussian_filter1d(data[key],
    #                                   sigma=3,
    #                                   axis=1)
    #
    #
    # dim = 2
    # sample = 50
    # data['velocity'] /= 20
    # stuff = ['torque', 'acceleration', 'velocity']
    # for key in stuff:
    #     plt.plot(data[key][sample, 1000:2000, dim])
    #
    # plt.legend(stuff)
    #
    # # plt.ylim([-10, 10])
    #
    # plt.show()
    # ipdb.set_trace()

    data = preprocess_data(data=data,
                           desired_n_data_points=10000,
                           smoothing_sigma=1)

    sys_id_lmi(robot=robot,
               angle=data['angle'],
               velocity=data['velocity'],
               acceleration=data['acceleration'],
               torque=data['torque'])

    # robot.simulate(dt=0.001, n_steps=10000)


def check_inertias():
    robot = Robot()

    inertia_matrix = robot.get_inertia_matrix_link_frame(0)

    # ipdb.set_trace()


def save_simulated_data(angles, velocities, torques, filename):
    """Stores the simulated data in a compatible format with the dynamics
       learning code."""
    data_dict = {}
    data_dict['measured_angles'] = angles
    data_dict['measured_velocities'] = velocities
    data_dict['measured_torques'] = torques
    data_dict['constrained_torques'] = torques
    data_dict['desired_torques'] = torques
    np.savez(filename, **data_dict)


def test_numeric_differentiation():
    dt = 0.001
    robot = Robot()

    data = {}
    data['angle'], data['velocity'], data['acceleration'], data['torque'] = robot.simulate(
        dt=dt, n_steps=10000)

    for key in data.keys():
        data[key] = np.expand_dims(data[key], axis=0)

    data_copy = data.copy()
    compute_accelerations(data_copy, dt=dt)

    for key in data.keys():
        difference = data[key][:,:-1] - data_copy[key]
        assert((np.absolute(difference) < 1e-12).all())


# if __name__ == '__main__':
#     try:
#         test_numeric_differentiation()
#     except:
#         extype, value, tb = sys.exc_info()
#         traceback.print_exc()
#         ipdb.post_mortem(tb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='system id baseline')
    parser.add_argument("--input", help="Filename of the input robot data")
    parser.add_argument("--output",
                        help="Filename to save simulated robot data")
    parser.add_argument("--visualizer", choices=['meshcat', 'gepetto'])
    args = parser.parse_args()
    robot = Robot()
    print(robot.model.inertias[2])
    if args.visualizer:
        if args.input:
            data = load_data()

            # Playing the first recorded angle trajectory.
            q_trajectory = np.matrix(data['angle'][0]).T
        else:

            # Playing artificial angle trajectory. Each degree of freedom is
            # linearly increased from 0 to PI independently.
            nsteps = 1000
            linear= np.linspace(0, np.pi, nsteps)
            zeros = np.zeros(nsteps)
            q_trajectory = np.block([[linear, zeros, zeros],
                                     [zeros, linear, zeros],
                                     [zeros, zeros, linear]])
            q_trajectory = np.matrix(q_trajectory)
        show_angle_trajectory(q_trajectory)
    if args.output:
        assert args.input
        data = load_data()
        nseq = data['angle'].shape[0]
        qs = []
        qdots = []
        taus = []
        for sample_idx in range(nseq):
            print(sample_idx)
            q, qdot, tau = robot.simulate(dt=0.001,
                                          torque=data['torque'][sample_idx],
                                          initial_angle=data['angle'][sample_idx, 0],
                                          initial_velocity=data['velocity'][sample_idx, 0])
            qs.append(np.expand_dims(q, 0))
            qdots.append(np.expand_dims(qdot, 0))
            taus.append(np.expand_dims(tau, 0))
        save_simulated_data(np.vstack(qs), np.vstack(qdots),
                            np.vstack(taus), args.output)

    # check_inertias()
    # test_sys_id_lmi()
    # test_sys_id_visually()
    # test_sys_id_simulated_torques()
