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
                   torque=data['torque'],
                   method_name='ls')
        elif self.identification_method == 'ls-lmi':
            sys_id(robot=self.robot,
                   angle=data['angle'],
                   velocity=data['velocity'],
                   acceleration=data['acceleration'],
                   torque=data['torque'],
                   method_name='lmi')
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
        self.viscous_friction = to_matrix(np.zeros(3)) + 0.0001
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
        return np.array(simulated_angles), np.array(simulated_vels), \
                np.array(simulated_accelerations), np.array(applied_torques)

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

        # TODO: Figure out why this fails some times.
        # just as a sanity check -----------------------------------------------
        Y = self.compute_regressor_matrix(angle, velocity, acceleration)
        actuator_torque_1 = Y * self.get_params()
        assert ((abs(actuator_torque - actuator_torque_1) <= 1e-6).all())
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

    def params_to_inertia_about_origin(self, params, link_index):
        if isinstance(params, np.ndarray):
            params = np.array(params).flatten()

        inertia_about_origin = \
            np.array([[params[10 * link_index + 4], params[10 * link_index + 5], params[10 * link_index + 7]],
                      [params[10 * link_index + 5], params[10 * link_index + 6], params[10 * link_index + 8]],
                      [params[10 * link_index + 7], params[10 * link_index + 8], params[10 * link_index + 9]]])
        return inertia_about_origin

    # see Wensing et al 2018 for details
    def params_to_second_moment(self, params, link_index):
        inertia_about_origin = self.params_to_inertia_about_origin(params, link_index)

        second_moment = np.diag([0.5 * np.trace(inertia_about_origin)
                                 for _ in range(3)]) - inertia_about_origin

        return second_moment

    # see Wensing et al 2018 for details
    def params_to_pseudo_inertia(self, params, link_index):
        second_moment = self.params_to_second_moment(params, link_index)
        mass_times_com = self.params_to_mass_times_com(params, link_index)
        mass = self.params_to_mass(params, link_index)

        pseudo_inertia = np.empty(shape=[4, 4], dtype=second_moment.dtype)

        pseudo_inertia[:3, :3] = second_moment
        pseudo_inertia[3, :3] = mass_times_com
        pseudo_inertia[:3, 3] = mass_times_com
        pseudo_inertia[3, 3] = mass

        return pseudo_inertia

    def params_to_mass_times_com(self, params, link_index):
        if isinstance(params, np.ndarray):
            params = np.array(params).flatten()

        mass_times_com = np.array([params[10 * link_index + 1],
                                   params[10 * link_index + 2], params[10 * link_index + 3]])
        return mass_times_com

    def params_to_mass(self, params, link_index):
        if isinstance(params, np.ndarray):
            params = np.array(params).flatten()

        mass = params[10 * link_index]
        return mass

    def params_to_viscous_friction(self, params, link_index):
        if isinstance(params, np.ndarray):
            params = np.array(params).flatten()

        return params[10 * self.count_degrees_of_freedom() + link_index]

    def params_to_static_friction(self, params, link_index):
        if isinstance(params, np.ndarray):
            params = np.array(params).flatten()

        return params[11 * self.count_degrees_of_freedom() + link_index]

    def count_degrees_of_freedom(self):
        return self.nv

    # getters and setters ------------------------------------------------------

    def get_params(self):
        theta = [self.model.inertias[i].toDynamicParameters()
                 for i in range(1, len(self.model.inertias))]

        theta = theta + [self.viscous_friction, self.static_friction]

        theta = np.concatenate(theta, axis=0)

        # some sanity checks
        for i in range(len(self.model.inertias) - 1):
            A = self.params_to_inertia_about_origin(theta, i)
            B = self.get_inertia_about_origin(i)
            assert(np.allclose(A, B))

            A = self.params_to_mass_times_com(theta, i)
            B = self.get_mass_times_com(i)
            assert(np.allclose(A, B))

            A = self.params_to_mass(theta, i)
            B = self.get_mass(i)
            assert(np.allclose(A, B))

            A = self.params_to_viscous_friction(theta, i)
            B = self.get_viscous_friction(i)
            assert(np.allclose(A, B))

            A = self.params_to_static_friction(theta, i)
            B = self.get_static_friction(i)
            assert(np.allclose(A, B))

            A = self.params_to_second_moment(theta, i)
            B = self.get_second_moment(i)
            assert(np.allclose(A, B))

        return theta

    def get_com(self, link_index):
        return np.array(self.model.inertias[link_index + 1].lever).flatten()

    def get_mass(self, link_index):
        return self.model.inertias[link_index + 1].mass

    def get_mass_times_com(self, link_index):
        return self.get_mass(link_index) * self.get_com(link_index)

    def get_inertia_about_com(self, link_index):
        return np.array(self.model.inertias[link_index + 1].inertia)

    def get_inertia_about_origin(self, link_index):
        inertia_matrix_com = self.get_inertia_about_com(link_index)
        com = self.get_com(link_index)
        mass = self.get_mass(link_index)

        # parallel axis theorem
        inertia_matrix_origin = inertia_matrix_com + mass * \
            (np.inner(com, com)*np.identity(3) - np.outer(com, com))
        return inertia_matrix_origin

    def get_viscous_friction(self, link_index):
        return self.viscous_friction[link_index]

    def get_static_friction(self, link_index):
        return self.static_friction[link_index]

    def get_second_moment(self, link_index):
        inertia_about_com = self.get_inertia_about_com(link_index)
        mass = self.get_mass(link_index)
        com = self.get_com(link_index)

        second_moment = 0.5 * np.trace(inertia_about_com) * \
            np.identity(3) - \
            inertia_about_com + mass * np.outer(com, com)

        return second_moment

    def set_params(self, theta):

        for dof in range(self.model.nv):
            theta_dof = theta[dof * 10: (dof + 1) * 10]

            self.model.inertias[dof + 1] = pinocchio.libpinocchio_pywrap.Inertia.FromDynamicParameters(
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

    def test_regressor_matrix(self):
        for _ in range(100):
            angle = pinocchio.randomConfiguration(self.model)
            velocity = pinocchio.utils.rand(self.model.nv)
            acceleration = pinocchio.utils.rand(self.model.nv)

            Y = self.compute_regressor_matrix(angle, velocity, acceleration)
            theta = self.get_params()
            other_tau = Y * theta

            torque = self.inverse_dynamics(angle, velocity, acceleration)

            assert ((abs(torque - other_tau) <= 1e-9).all())


def compute_accelerations(data, dt):
    data['acceleration'] = np.diff(data['velocity'], axis=1) / dt
    for key in ['angle', 'velocity', 'torque']:
        data[key] = data[key][:, :-1]

    # test that everything worked out -----------------------------------------
    integrated_velocity = data['velocity'][:, :-1] + \
        data['acceleration'][:, :-1] * dt

    is_consistent = (np.absolute(integrated_velocity -
                                 data['velocity'][:, 1:]) <= 1e-12).all()
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
    return np.squeeze(np.array(np.sqrt(
        (Y * theta - T).transpose().dot(Y * theta - T) / len(T))))


def check_and_log(log, robot, angle, velocity, acceleration, torque, Y, T, suffix):
    log['rank Y ' + suffix] = np.linalg.matrix_rank(Y)

    rmse_a = rmse_sequential(robot=robot,
                             angle=angle,
                             velocity=velocity,
                             acceleration=acceleration,
                             torque=torque)
    rmse_b = rmse_batch(theta=robot.get_params(),
                        Y=Y, T=T)
    assert(abs(rmse_a - rmse_b) <= 1e-6)
    log['rmse ' + suffix] = rmse_a

    for i in range(robot.count_degrees_of_freedom()):
        inertia = robot.get_inertia_about_com(i)
        eigenvalues, eigenvectors = np.linalg.eig(inertia)
        reconstruction = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.transpose())
        assert(np.allclose(reconstruction, inertia, atol=1e-6))
        assert(np.allclose(eigenvectors.dot(eigenvectors.transpose()), np.identity(3), atol=1e-6))

        log['params ' + suffix] = np.array(robot.get_params()).flatten()
        log['eigenvalues of inertia ' + str(i) + ' ' + suffix] = eigenvalues
        log['mass ' + str(i) + ' ' + suffix] = robot.get_mass(i)
        log['com ' + str(i) + ' ' + suffix] = robot.get_com(i)
        log['static friction  ' + str(i) + ' ' + suffix] = robot.get_static_friction(i)
        log['viscous friction  ' + str(i) + ' ' + suffix] = robot.get_viscous_friction(i)


def sys_id(robot, angle, velocity, acceleration, torque, method_name):
    log = dict()
    robot.test_regressor_matrix()

    Y = np.concatenate(
        [robot.compute_regressor_matrix(angle[t], velocity[t], acceleration[t])
         for t in
         range(angle.shape[0])], axis=0)

    T = np.concatenate(
        [to_matrix(torque[t]) for t in range(angle.shape[0])], axis=0)

    check_and_log(log=log, robot=robot, angle=angle, velocity=velocity,
                  acceleration=acceleration, torque=torque, Y=Y, T=T, suffix='before id')

    if method_name == 'lmi':
        theta = sys_id_lmi(robot=robot, Y=Y, T=T)
    elif method_name == 'ls':
        theta = sys_id_ls(robot=robot, Y=Y, T=T)
    else:
        raise NotImplementedError

    robot.set_params(theta)

    check_and_log(log=log, robot=robot, angle=angle, velocity=velocity,
                  acceleration=acceleration, torque=torque, Y=Y, T=T, suffix='after id')

    for key in sorted(log.keys()):
        print(key + ': ', log[key], '\n')


def sys_id_lmi(robot, Y, T):
    theta = cvxpy.Variable((36, 1))  # [m, mc_x, mc_y, mc_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]
    theta_cad = np.asarray(robot.get_params())
    # it is not clear whether norm or sum of squares is better for solver
    cost = cvxpy.sum_squares(Y * theta - T) + 1e-6 * cvxpy.sum_squares(theta - theta_cad)

    pseudo_inertias = []
    static_frictions = []
    viscous_frictions = []
    masses = []
    constraints = []
    for i in range(robot.count_degrees_of_freedom()):
        pseudo_inertias += [cvxpy.bmat(robot.params_to_pseudo_inertia(theta, i))]
        static_frictions += [robot.params_to_static_friction(theta, i)]
        viscous_frictions += [robot.params_to_viscous_friction(theta, i)]
        masses += [robot.params_to_mass(theta, i)]

        constraints += [masses[i] >= 0.01]
        constraints += [masses[i] <= 0.5]
        constraints += [pseudo_inertias[i] >> 0]
        constraints += [static_frictions[i] >= 0]
        constraints += [viscous_frictions[i] >= 0]

    problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    theta.value = theta_cad
    problem.solve(solver='MOSEK', warm_start=True)
    assert(all(c.value() for c in constraints))
    assert(theta.value is not None)

    return np.array(theta.value)


def sys_id_ls(robot, Y, T):
    regularization_epsilon = 1e-10
    regularization_mu = np.asarray(robot.get_params())
    theta = np.linalg.solve(
        Y.transpose() * Y + regularization_epsilon * np.eye(Y.shape[1],
                                                            Y.shape[1]),
        Y.transpose() * T + regularization_epsilon * regularization_mu)

    return theta


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
        difference = data[key][:, :-1] - data_copy[key]
        assert((np.absolute(difference) < 1e-12).all())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='system id baseline')
    parser.add_argument("--input", help="Filename of the input robot data")
    parser.add_argument("--output",
                        help="Filename to save simulated robot data")
    parser.add_argument("--visualizer", choices=['meshcat', 'gepetto'])
    parser.add_argument("--noise", type=float)
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
            linear = np.linspace(0, np.pi, nsteps)
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
            q, qdot, _, tau = robot.simulate(dt=0.001,
                    torque=data['torque'][sample_idx],
                    initial_angle=data['angle'][sample_idx, 0],
                    initial_velocity=data['velocity'][sample_idx, 0])
            qs.append(np.expand_dims(q, 0))
            qdots.append(np.expand_dims(qdot, 0))
            taus.append(np.expand_dims(tau, 0))

        qs = np.vstack(qs)
        qdots = np.vstack(qdots)
        taus = np.vstack(taus)
        if args.noise:
            qs = qs + args.noise * np.random.randn(*qs.shape)
            qdots = qdots + args.noise * np.random.randn(*qdots.shape)
            taus = taus + args.noise * np.random.randn(*taus.shape)
        save_simulated_data(qs, qdots, taus, args.output)

    # check_inertias()
    # test_sys_id_lmi()
    # test_sys_id_visually()
    # test_sys_id_simulated_torques()
