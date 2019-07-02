from __future__ import print_function

import sys
import ipdb
import traceback

import numpy as np
import numpy.matlib
import time

import matplotlib.pylab as plt

import pinocchio

from pinocchio.robot_wrapper import RobotWrapper
from os.path import join, dirname

from scipy.ndimage import gaussian_filter1d

import rospkg

# dynamics learning stuff
from DL import DynamicsLearnerInterface


class SystemId(DynamicsLearnerInterface):
    def __init__(self,
                 history_length,
                 prediction_horizon,
                 averaging):
        DynamicsLearnerInterface.__init__(self,
                                          history_length=history_length,
                                          prediction_horizon=prediction_horizon,
                                          difference_learning=False,
                                          averaging=averaging,
                                          streaming=False)

        if averaging:
            raise NotImplementedError

        self.robot = Robot()

    def learn(self, observation_sequences, action_sequences):
        ### TODO: here we assume that the observations and actions
        ### have a specific order. furthermore, we ignore the measured torques,
        ### we only take the commanded torques

        data = dict()
        data['angle'] = observation_sequences[:, :, :3]
        data['velocity'] = observation_sequences[:, :, 3:6]
        data['torque'] = action_sequences

        ### TODO: adjust parameters
        print('identifying....')
        sys_id(self.robot, *preprocess_data(data=data,
                                            desired_n_data_points=100000,
                                            smoothing_sigma=1))

    def predict(self, observation_history, action_history, action_future=None):
        if action_future is None:
            assert self.prediction_horizon == 1
            action_future = np.empty((observation_history.shape[0],
                                      0,
                                      self.action_dimension))
        assert (action_future.shape[1] == self.prediction_horizon - 1)

        n_samples = observation_history.shape[0]
        dim_observation = observation_history.shape[2]

        action_present_future = np.append(action_history[:, -1:],
                                          action_future,
                                          axis=1)

        predictions = np.empty((n_samples, dim_observation))
        predictions[:] = numpy.nan

        for i in xrange(n_samples):
            angle = observation_history[i, -1, :3]
            velocity = observation_history[i, -1, 3:6]
            torques_sequence = action_present_future[i]

            integration_step_ms = max(self.prediction_horizon / 10, 1)

            for t in xrange(0, self.prediction_horizon, integration_step_ms):
                angle, velocity = \
                    self.robot.predict(angle,
                                       velocity,
                                       torques_sequence[t],
                                       integration_step_ms / 1000.)

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
    def __init__(self):
        self.load_urdf()
        self.viscous_friction = to_matrix(np.zeros(3)) + 0.01
        self.static_friction = to_matrix(np.zeros(3))

    # dynamics -----------------------------------------------------------------
    def simulate(self, n_seconds=10):
        self.initViewer(loadModel=True)

        # angle = pinocchio.randomConfiguration(robot.model)
        angle = np.transpose(np.matrix([1.0, -1.35, 3.0]))
        velocity = pinocchio.utils.zero(self.model.nv)
        torque = pinocchio.utils.zero(self.model.nv)

        dt = 0.01
        for _ in xrange(int(n_seconds / dt)):
            acceleration = self.forward_dynamics(angle, velocity, torque)

            angle = angle + velocity * dt
            velocity = velocity + acceleration * dt

            self.display(angle)
            time.sleep(dt)

    # TODO: this needs to be checked
    def predict(self, angle, velocity, torque, dt):
        angle = to_matrix(angle)
        velocity = to_matrix(velocity)
        torque = to_matrix(torque)
        acceleration = self.forward_dynamics(angle, velocity, torque)
        angle = angle + velocity * dt
        velocity = velocity + acceleration * dt

        return angle, velocity

    def friction_torque(self, velocity):
        return -(np.multiply(velocity, self.viscous_friction) +
                 np.multiply(np.sign(velocity), self.static_friction))

    def forward_dynamics(self, angle, velocity, actuator_torque):
        joint_torque = actuator_torque + self.friction_torque(velocity)

        return pinocchio.aba(self.model, self.data, angle, velocity, joint_torque)

    def inverse_dynamics(self, angle, velocity, acceleration):

        joint_torque = pinocchio.rnea(self.model, self.data, angle, velocity, acceleration)
        actuator_torque = joint_torque - self.friction_torque(velocity)

        return actuator_torque

    def compute_regressor_matrix(self, angle, velocity, acceleration):
        joint_torque_regressor = \
            pinocchio.computeJointTorqueRegressor(self.model, self.data,
                                                  to_matrix(angle),
                                                  to_matrix(velocity),
                                                  to_matrix(acceleration))

        viscous_friction_torque_regressor = to_diagonal_matrix(velocity)
        static_friction_torque_regressor = to_diagonal_matrix(np.sign(velocity))

        regressor_matrix = np.concatenate([
            joint_torque_regressor,
            viscous_friction_torque_regressor,
            static_friction_torque_regressor], axis=1)

        return regressor_matrix

    # getters and setters ------------------------------------------------------
    def get_params(self):
        theta = [self.model.inertias[i].toDynamicParameters()
                 for i in xrange(1, len(self.model.inertias))]

        theta = theta + [self.viscous_friction, self.static_friction]

        theta = np.concatenate(theta, axis=0)
        return theta

    def set_params(self, theta):
        for dof in xrange(self.model.nv):
            theta_dof = theta[dof * 10: (dof + 1) * 10]
            self.model.inertias[dof + 1] = \
                pinocchio.libpinocchio_pywrap.Inertia.FromDynamicParameters(
                    theta_dof)

        n_inertial_params = self.model.nv * 10
        self.viscous_friction = theta[n_inertial_params: n_inertial_params + 3]
        self.static_friction = theta[
                               n_inertial_params + 3: n_inertial_params + 6]

        assert (((self.get_params() - theta) < 1e-9).all())

    # loading ------------------------------------------------------------------
    def load_urdf(self):
        urdf_path = (
            join(rospkg.RosPack().get_path("robot_properties_manipulator"),
                 "urdf",
                 "manipulator.urdf"))
        meshes_path = [
            dirname(rospkg.RosPack().get_path("robot_properties_manipulator"))]

        self.initFromURDF(urdf_path, meshes_path)


def test(robot):
    angle = pinocchio.randomConfiguration(robot.model)
    velocity = pinocchio.utils.rand(robot.model.nv)
    acceleration = pinocchio.utils.rand(robot.model.nv)

    Y = robot.compute_regressor_matrix(angle, velocity, acceleration)
    theta = robot.get_params()
    other_tau = Y * theta

    torque = robot.inverse_dynamics(angle, velocity, acceleration)

    assert ((abs(torque - other_tau) <= 1e-9).all())


def load_and_preprocess_data(desired_n_data_points=10000):
    all_data = np.load('/is/ei/mwuthrich/dataset_v06_sines_full.npz')

    data = dict()
    data['angle'] = all_data['measured_angles']
    data['velocity'] = all_data['measured_velocities']
    ### TODO: not sure whether to take measured or constrained torques
    data['torque'] = all_data['measured_torques']

    return preprocess_data(data, desired_n_data_points)


def preprocess_data(data, desired_n_data_points, smoothing_sigma=0.0001):
    data['acceleration'] = np.diff(data['velocity'], axis=1)
    for key in ['angle', 'velocity', 'torque']:
        data[key] = data[key][:, :-1]

    # smoothen -----------------------------------------------------------------
    for key in data.keys():
        data['smooth_' + key] = \
            gaussian_filter1d(data[key],
                              sigma=smoothing_sigma,
                              axis=1)

    # cut off ends -------------------------------------------------------------
    for key in data.keys():
        data[key] = data[key][:, 1000: -1000]

    # plot ---------------------------------------------------------------------
    # dim = 2
    # sample = 100
    # for key in ['torque']:
    #     plt.plot(data[key][sample, :, dim])
    #     plt.plot(data['smooth_' + key][sample, :, dim])
    # plt.show()

    # reshape ------------------------------------------------------------------
    n_data_points = data['angle'].shape[0] * data['angle'].shape[1]
    test = np.copy(data['angle'])
    for key in data.keys():
        data[key] = np.reshape(data[key], [n_data_points, 3])

    assert ((test[23, 1032] == data['angle'][23 * test.shape[1] + 1032]).all())

    # return a random subset of the datapoints ---------------------------------
    data_point_indices = \
        np.random.permutation(np.arange(n_data_points))[:desired_n_data_points]

    for key in data.keys():
        data[key] = data[key][data_point_indices]

    return data['smooth_angle'], data['smooth_velocity'], \
           data['smooth_acceleration'], data['smooth_torque']


def satisfies_normal_equation(theta, Y, T, epsilon=1e-6):
    lhs = (Y.transpose() * Y).dot(theta)
    rhs = Y.transpose().dot(T)
    return (abs(lhs - rhs) < epsilon).all()


def sys_id(robot, angle, velocity, acceleration, torque):
    Y = np.concatenate(
        [robot.compute_regressor_matrix(angle[t], velocity[t], acceleration[t]) for t in
         xrange(angle.shape[0])], axis=0)

    T = np.concatenate(
        [to_matrix(torque[t]) for t in xrange(angle.shape[0])], axis=0)

    regularization_epsilon = 1e-12
    regularization_mu = np.matrix(np.zeros(Y.shape[1]) + 1e-6).transpose()
    theta = np.linalg.solve(
        Y.transpose() * Y + regularization_epsilon * np.eye(Y.shape[1],
                                                            Y.shape[1]),
        Y.transpose() * T + regularization_epsilon * regularization_mu)

    robot.set_params(theta)

    assert (satisfies_normal_equation(robot.get_params(), Y, T))


if __name__ == '__main__':
    try:
        robot = Robot()
        test(robot)

        sys_id(robot, *load_and_preprocess_data())
    except:
        traceback.print_exc(sys.stdout)
        _, _, tb = sys.exc_info()
        ipdb.post_mortem(tb)
