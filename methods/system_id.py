from __future__ import print_function

import sys
import ipdb
import traceback

import numpy as np
import numpy.matlib
import time

import math

import matplotlib.pylab as plt

import pinocchio

from pinocchio.robot_wrapper import RobotWrapper
from os.path import join, dirname

from scipy.ndimage import gaussian_filter1d

import rospkg

import time

if sys.version_info[0] >= 3:
    import cvxpy

# dynamics learning stuff
from DL import DynamicsLearnerInterface


class SystemId(DynamicsLearnerInterface):
    def __init__(self,
                 history_length,
                 prediction_horizon,
                 identification_method):
        DynamicsLearnerInterface.__init__(self,
                                          history_length=history_length,
                                          prediction_horizon=prediction_horizon,
                                          difference_learning=False,
                                          averaging=False,
                                          streaming=False)
        self.robot = Robot()
        self.identification_method = identification_method

    def learn(self, observation_sequences, action_sequences):
        print('learning')
        # preprocess data ------------------------------------------------------
        data = dict()
        data['angle'] = observation_sequences[:, :, :3]
        data['velocity'] = observation_sequences[:, :, 3:6]
        data['torque'] = action_sequences
        data = compute_accelerations(data)
        data = preprocess_data(data=data,
                               desired_n_data_points=100000,
                               smoothing_sigma=1.0)

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
            raise NotImplementedError
        else:
            raise NotImplementedError


    def predict(self, observation_history, action_history, action_future=None):
        # parse arguments ------------------------------------------------------
        n_samples = observation_history.shape[0]
        dim_observation = observation_history.shape[2]

        if action_future is None:
            assert self.prediction_horizon == 1
            action_future = np.empty((n_samples, 0, self.action_dimension))

        assert (action_future.shape[1] == self.prediction_horizon - 1)

        # we smoothen velocities -----------------------------------------------
        temp = observation_history.copy()
        observation_history[..., 3:6] = gaussian_filter1d(
            observation_history[..., 3:6],
            sigma=2,
            axis=1)

        # make predictions -----------------------------------------------------
        predictions = np.empty((n_samples, dim_observation))
        predictions[:] = numpy.nan

        for i in xrange(n_samples):
            torques_sequence = np.append(action_history[i, -1:],
                                         action_future[i],
                                         axis=0)

            angle = observation_history[i, -1, :3]
            velocity = observation_history[i, -1, 3:6]

            integration_step_ms = 1
            # for long horizon we increase the
            # integration step to get a speedup
            if self.prediction_horizon == 1000:
                integration_step_ms = 10

            for t in xrange(0, self.prediction_horizon, integration_step_ms):
                torque = np.average(
                    torques_sequence[t: t + integration_step_ms], axis=0)
                angle, velocity = \
                    self.robot.predict(angle=angle,
                                       velocity=velocity,
                                       torque=torque,
                                       dt=integration_step_ms / 1000.)

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
        self.static_friction = to_matrix(np.zeros(3)) + 0.00

    # dynamics -----------------------------------------------------------------
    def simulate(self,
                 dt,
                 n_steps=None,
                 torque=None,
                 initial_angle=None,
                 initial_velocity=None,
                 mask=np.ones(3),
                 verbose=False):
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

        self.initViewer(loadModel=True)
        last_time = time.time()
        for t in xrange(torque.shape[0]):
            acceleration = self.forward_dynamics(angle, velocity, torque[t])

            angle = angle + np.multiply(mask, velocity * dt)
            velocity = velocity + np.multiply(mask, acceleration * dt)

            self.display(angle)
            if verbose:
                print('angle: ', np.array(angle).flatten(),
                      '\nvelocity: ', np.array(velocity).flatten())

            sleep_time = last_time + dt - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            current_time = time.time()
            if verbose:
                print('time elapsed in cycle', current_time - last_time)
            last_time = current_time

    def show_trajectory(self, dt, angle):
        self.initViewer(loadModel=True)

        last_time = time.time()
        for t in xrange(angle.shape[0]):
            self.display(to_matrix(angle[t]))

            sleep_time = last_time + dt - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            current_time = time.time()
            print('time elapsed in cycle', current_time - last_time)
            last_time = current_time

            print(angle[t])

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


def test_regressor_matrix(robot):
    for _ in xrange(100):
        angle = pinocchio.randomConfiguration(robot.model)
        velocity = pinocchio.utils.rand(robot.model.nv)
        acceleration = pinocchio.utils.rand(robot.model.nv)

        Y = robot.compute_regressor_matrix(angle, velocity, acceleration)
        theta = robot.get_params()
        other_tau = Y * theta

        torque = robot.inverse_dynamics(angle, velocity, acceleration)

        assert ((abs(torque - other_tau) <= 1e-9).all())


def load_data():
    all_data = np.load('/is/ei/mwuthrich/dataset_v06_sines_full.npz')

    data = dict()
    data['angle'] = all_data['measured_angles']
    data['velocity'] = all_data['measured_velocities']
    ### TODO: not sure whether to take measured or constrained torques
    data['torque'] = all_data['constrained_torques']

    # robot = Robot()
    # sample_idx = 113
    # robot.show_trajectory(dt=0.001,
    #                       angle=data['angle'][sample_idx, :2000])
    #
    # robot.simulate(dt=0.001,
    #                torque=data['torque'][sample_idx, :2000],
    #                initial_angle=data['angle'][sample_idx, 0],
    #                initial_velocity=data['velocity'][sample_idx, 0])

    return data


def compute_accelerations(data):
    data['acceleration'] = np.diff(data['velocity'], axis=1)
    for key in ['angle', 'velocity', 'torque']:
        data[key] = data[key][:, :-1]

    return data


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
                    data[key].shape[1] / 10: -data[key].shape[1] / 10]

    # plot ---------------------------------------------------------------------
    # dim = 2
    # sample = 100
    # for key in ['torque']:
    #     plt.plot(data[key][sample, :, dim])
    #     plt.plot(data['smooth_' + key][sample, :, dim])
    # plt.show()

    # reshape ------------------------------------------------------------------
    ordered_data = data.copy()
    n_trajectories = data['angle'].shape[0]
    n_time_steps = data['angle'].shape[1]
    n_data_points = n_trajectories * n_time_steps
    for key in data.keys():
        data[key] = np.reshape(data[key], [n_data_points, 3])

    for _ in xrange(10):
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

    for t in xrange(T):
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
         xrange(angle.shape[0])], axis=0)

    T = np.concatenate(
        [to_matrix(torque[t]) for t in xrange(angle.shape[0])], axis=0)

    log['rmse_sequential_before_id'] = rmse_sequential(robot=robot,
                                                       angle=angle,
                                                       velocity=velocity,
                                                       acceleration=acceleration,
                                                       torque=torque)
    log['rmse_batch_before_id'] = rmse_batch(theta=robot.get_params(),
                                             Y=Y, T=T)


    #Constrained optimization using CVXPY
    #theta = [m, mc_x, mc_y, mc_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]
    theta = cvxpy.Variable((36, 1))

    cost = Y*theta - T
    cost = cvxpy.norm(cost)

    J = []
    constraints = []
    for i in range(3):
        J += [cvxpy.bmat([[0.5*(-theta[i*12+4]+theta[i*12+6]+theta[i*12+9]), -theta[i*12+5], -theta[i*12+7], theta[i*12+1]],
                            [-theta[i*12+5], 0.5*(theta[i*12+4]-theta[i*12+6]+theta[i*12+9]), -theta[i*12+8], theta[i*12+2]],
                            [-theta[i*12+7], -theta[i*12+8], 0.5*(theta[i*12+4]+theta[i*12+6]-theta[i*12+9]), theta[i*12+3]],
                            [theta[i*12+1], theta[i*12+2], theta[i*12+3], theta[i*12]]])]
        constraints += [J[i] >> 0]

    # mass
    constraints += [theta[0] >= 0.1]
    constraints += [theta[0] <= 1]
    constraints += [theta[12] >= 0.1]
    constraints += [theta[12] <= 1]
    constraints += [theta[24] >= 0.03]
    constraints += [theta[24] <= 1]
    # friction coeffs
    constraints += [theta[10] >= 0]
    constraints += [theta[11] >= 0]
    constraints += [theta[22] >= 0]
    constraints += [theta[23] >= 0]
    constraints += [theta[34] >= 0]
    constraints += [theta[35] >= 0]
    # COM*mass, assumes an upper bound of 1kg on the link mass
    constraints += [theta[1] <= 0.2]
    constraints += [theta[1] >= -0.2]
    constraints += [theta[2] <= 0.2]
    constraints += [theta[2] >= -0.2]
    constraints += [theta[3] <= 0.2]
    constraints += [theta[3] >= -0.2]
    constraints += [theta[13] <= 0.2]
    constraints += [theta[13] >= -0.2]
    constraints += [theta[14] <= 0.2]
    constraints += [theta[14] >= -0.2]
    constraints += [theta[15] <= 0.2]
    constraints += [theta[15] >= -0.2]
    constraints += [theta[25] <= 0.2]
    constraints += [theta[25] >= -0.2]
    constraints += [theta[26] <= 0.2]
    constraints += [theta[26] >= -0.2]
    constraints += [theta[27] <= 0.2]
    constraints += [theta[28] >= -0.2]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(verbose=False,eps=10e-8,max_iters=20000,solver='SCS')

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
    log['rmse_batch_after_id_lmi'] = rmse_batch(theta=robot.get_params(), Y=Y, T=T)

    for key in log.keys():
        print(key + ': ', log[key], '\n')

def sys_id(robot, angle, velocity, acceleration, torque):
    log = dict()

    test_regressor_matrix(robot)

    Y = np.concatenate(
        [robot.compute_regressor_matrix(angle[t], velocity[t], acceleration[t])
         for t in
         xrange(angle.shape[0])], axis=0)

    T = np.concatenate(
        [to_matrix(torque[t]) for t in xrange(angle.shape[0])], axis=0)

    log['rmse_sequential_before_id'] = rmse_sequential(robot=robot,
                                                       angle=angle,
                                                       velocity=velocity,
                                                       acceleration=acceleration,
                                                       torque=torque)
    log['rmse_batch_before_id'] = rmse_batch(theta=robot.get_params(),
                                             Y=Y, T=T)

    regularization_epsilon = 1e-10
    regularization_mu = robot.get_params()
    theta = np.linalg.solve(
        Y.transpose() * Y + regularization_epsilon * np.eye(Y.shape[1],
                                                            Y.shape[1]),
        Y.transpose() * T + regularization_epsilon * regularization_mu)

    log['rmse_batch_optimal_theta_after_id'] = rmse_batch(theta=theta,
                                                          Y=Y, T=T)

    robot.set_params(theta)

    assert (satisfies_normal_equation(robot.get_params(), Y, T))
    test_regressor_matrix(robot)

    log['rmse_sequential_after_id'] = rmse_sequential(robot=robot,
                                                      angle=angle,
                                                      velocity=velocity,
                                                      acceleration=acceleration,
                                                      torque=torque)
    log['rmse_batch_after_id'] = rmse_batch(theta=robot.get_params(), Y=Y, T=T)

    for key in log.keys():
        print(key + ': ', log[key], '\n')


def test_sys_id_simumlated_torques():
    robot = Robot()
    test_regressor_matrix(robot)

    # create dataset with simulated torques ------------------------------------
    data = load_data()
    data = compute_accelerations(data)
    for key in data.keys():
        data[key] = data[key][:, 10000: 10200]

    data['torque'] = [[robot.inverse_dynamics(
        angle=data['angle'][trajectory_idx, t],
        velocity=data['velocity'][trajectory_idx, t],
        acceleration=data['acceleration'][trajectory_idx, t])
        for t in xrange(data['angle'].shape[1])]
        for trajectory_idx in xrange(data['angle'].shape[0])]
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
    robot = Robot()
    # robot.simulate(dt=0.001,
    #                n_steps=1000,
    #                torque=[0.1, 0.1, 0.1],
    #                initial_angle=[1, 1, 1],
    #                mask=[1, 1, 1])

    # robot.simulate(dt=0.001, n_steps=10000)

    data = load_data()
    data = compute_accelerations(data)
    data = preprocess_data(data=data,
                           desired_n_data_points=10000,
                           smoothing_sigma=None)
    sys_id(robot=robot,
           angle=data['angle'],
           velocity=data['velocity'],
           acceleration=data['acceleration'],
           torque=data['torque'])

    # robot.simulate(dt=0.001, n_steps=10000)


if __name__ == '__main__':
    try:
        test_sys_id_simumlated_torques()
        test_sys_id_visually()

    except:
        traceback.print_exc(sys.stdout)
        _, _, tb = sys.exc_info()
        ipdb.post_mortem(tb)
