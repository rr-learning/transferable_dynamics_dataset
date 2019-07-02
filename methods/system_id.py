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

        # q = pinocchio.randomConfiguration(robot.model)
        q = np.transpose(np.matrix([1.0, -1.35, 3.0]))
        v = pinocchio.utils.zero(self.model.nv)
        tau = pinocchio.utils.zero(self.model.nv)

        dt = 0.01
        for _ in xrange(int(n_seconds / dt)):
            a = self.forward_dynamics(q, v, tau)

            q = q + v * dt
            v = v + a * dt

            print(q)

            self.display(q)
            time.sleep(dt)

    # TODO: this needs to be checked
    def predict(self, q, v, tau_horizon, dt):
        for t in xrange(tau_horizon.shape[0]):
            a = self.forward_dynamics(q, v, tau_horizon[t])
            q = q + v * dt
            v = v + a * dt

        return q, v, tau_horizon[-1]


    def friction_torque(self, v):
        return -(np.multiply(v, self.viscous_friction) +
                 np.multiply(np.sign(v), self.static_friction))

    def forward_dynamics(self, qq, v, actuator_torque):
        joint_torque = actuator_torque + self.friction_torque(v)

        return pinocchio.aba(self.model, self.data, qq, v, joint_torque)

    def inverse_dynamics(self, qq, v, a):

        joint_torque = pinocchio.rnea(self.model, self.data, qq, v, a)
        actuator_torque = joint_torque - self.friction_torque(v)

        return actuator_torque

    def compute_regressor_matrix(self, qq, v, a):
        joint_torque_regressor = \
            pinocchio.computeJointTorqueRegressor(self.model, self.data,
                                                  to_matrix(qq),
                                                  to_matrix(v),
                                                  to_matrix(a))

        viscous_friction_torque_regressor = to_diagonal_matrix(v)
        static_friction_torque_regressor = to_diagonal_matrix(np.sign(v))

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
    q = pinocchio.randomConfiguration(robot.model)
    v = pinocchio.utils.rand(robot.model.nv)
    a = pinocchio.utils.rand(robot.model.nv)

    Y = robot.compute_regressor_matrix(q, v, a)
    theta = robot.get_params()
    other_tau = Y * theta

    tau = robot.inverse_dynamics(q, v, a)

    assert ((abs(tau - other_tau) <= 1e-9).all())


def load_and_preprocess_data(desired_n_data_points=10000):
    all_data = np.load('/is/ei/mwuthrich/dataset_v06_sines_full.npz')

    data = dict()
    data['angles'] = all_data['measured_angles']
    data['velocities'] = all_data['measured_velocities']
    ### TODO: not sure whether to take measured or constrained torques
    data['torques'] = all_data['measured_torques']
    data['accelerations'] = np.diff(data['velocities'], axis=1)
    for key in ['angles', 'velocities', 'torques']:
        data[key] = data[key][:, :-1]

    # smoothen -----------------------------------------------------------------
    sigma = 5
    for key in data.keys():
        data['smooth_' + key] = gaussian_filter1d(data[key], sigma=5, axis=1)

    # cut off ends -------------------------------------------------------------
    for key in data.keys():
        data[key] = data[key][:, 1000 : -1000]

    # plot ---------------------------------------------------------------------
    # dim = 2
    # sample = 100
    # for key in ['torques']:
    #     plt.plot(data[key][sample, :, dim])
    #     plt.plot(data['smooth_' + key][sample, :, dim])
    # plt.show()

    # reshape ------------------------------------------------------------------
    n_data_points = data['angles'].shape[0] * data['angles'].shape[1]
    test = np.copy(data['angles'])
    for key in data.keys():
        data[key] = np.reshape(data[key], [n_data_points, 3])

    assert((test[23, 1032] == data['angles'][23 * test.shape[1] + 1032]).all())

    # return a random subset of the datapoints ---------------------------------
    data_point_indices = \
        np.random.permutation(np.arange(n_data_points))[:desired_n_data_points]

    for key in data.keys():
        data[key] = data[key][data_point_indices]

    return data['smooth_angles'], data['smooth_velocities'], \
           data['smooth_accelerations'], data['smooth_torques']


def satisfies_normal_equation(theta, Y, T, epsilon=1e-6):
    lhs = (Y.transpose() * Y).dot(theta)
    rhs = Y.transpose().dot(T)
    return (abs(lhs - rhs) < epsilon).all()


def sys_id(qq, v, a, tau):
    robot = Robot()

    Y = np.concatenate(
            [robot.compute_regressor_matrix(qq[t], v[t], a[t]) for t in
             xrange(qq.shape[0])], axis=0)

    T = np.concatenate(
        [to_matrix(tau[t]) for t in xrange(qq.shape[0])], axis=0)

    regularization_epsilon = 1e-12
    regularization_mu = np.matrix(np.zeros(Y.shape[1]) + 1e-6).transpose()
    theta = np.linalg.solve(
        Y.transpose() * Y + regularization_epsilon * np.eye(Y.shape[1],
                                                            Y.shape[1]),
        Y.transpose() * T + regularization_epsilon * regularization_mu)

    robot.set_params(theta)

    assert (satisfies_normal_equation(robot.get_params(), Y, T))

    robot.simulate(1000)


if __name__ == '__main__':
    try:
        robot = Robot()
        test(robot)

        sys_id(*load_and_preprocess_data())
    except:
        traceback.print_exc(sys.stdout)
        _, _, tb = sys.exc_info()
        ipdb.post_mortem(tb)
