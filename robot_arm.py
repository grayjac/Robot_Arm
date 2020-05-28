import numpy as np
from matplotlib import pyplot as plt
import itertools as it
from scipy.optimize import fmin


class RobotArm:

    def __init__(self, *arm_lengths, obstacles=None):
        """
        Represents an N-link arm with the arm lengths given.
        Example of initializing a 3-link robot with a single obstacle:

        my_arm = RobotArm(0.58, 0.14, 0.43, obstacles=[VerticalWall(0.45)])

        :param arm_lengths: Float values representing arm lengths of the robot.
        :param obstacles:
        """
        self.arm_lengths = np.array(arm_lengths)
        if np.any(self.arm_lengths < 0):
            raise ValueError("Cannot have negative arm length!")
        self.obstacles = []
        if obstacles is not None:
            self.obstacles = obstacles

    def __repr__(self):
        msg = '<RobotArm with {} links\nArm lengths: '.format(len(self.arm_lengths))
        msg += ', '.join(['{:.2f}'.format(length) for length in self.arm_lengths])
        msg += '\nObstacles: '
        if not len(self.obstacles):
            msg += 'None'
        else:
            msg += '\n\t' + '\n\t'.join(str(obstacle) for obstacle in self.obstacles)
        msg += '\n>'
        return msg

    def __str__(self):
        return self.__repr__()

    def get_links(self, thetas):
        """
        Returns all of the link locations of the robot as Link objects.
        :param thetas: A list or array of scalars matching the number of arms.
        :return: A list of Link objects.
        """

        cum_theta = np.cumsum(thetas)

        results = np.zeros((self.arm_lengths.shape[0] + 1, 2))

        results[1:, 0] = np.cumsum(self.arm_lengths * np.cos(cum_theta))
        results[1:, 1] = np.cumsum(self.arm_lengths * np.sin(cum_theta))
        links = [Link(start, end) for start, end in zip(results[:-1], results[1:])]

        return links

    def get_ee_location(self, thetas):
        """
        Returns the location of the end effector as a length 2 Numpy array.
        :param thetas: A list or array of scalars matching the number of arms.
        :return: A length 2 Numpy array of the x,y coordinate.
        """
        return self.get_links(thetas)[-1].end

    def ik_grid_search(self, target, intervals):

        dist_dict = {}

        for thetas in it.product(np.linspace(0, 2 * np.pi, intervals, endpoint=False), repeat=len(self.arm_lengths)):
            for i in range(len(self.arm_lengths)):
                if self.get_links(thetas)[i].check_any_wall_collision(self.obstacles):  # If any of the links collide
                    collision_indicator = True  # Boolean indicator. True if theres a collision, False if not
                    break
                else:
                    collision_indicator = False

            if not collision_indicator:  # Runs only if collision_indicator == True (no collision)
                dist = np.linalg.norm(target - self.get_ee_location(thetas))
                dist_dict.update({thetas: dist})

        return min(dist_dict, key=dist_dict.get), dist_dict[min(dist_dict, key=dist_dict.get)]

    def distance_from_target(self, thetas, target):
        dist = np.linalg.norm(target - self.get_ee_location(thetas))
        return dist

    def ik_fmin_search(self, target, thetas_guess, max_calls=100):

        opt = fmin(lambda array: self.distance_from_target(array, target),
                   [thetas_guess], maxfun=max_calls, full_output=True)
        return opt[0], opt[1], opt[3]

    def get_collision_score(self, thetas):
        """
        Returns the number of collisions each link has with each obstacle in the robot arm's environment.
        :param thetas: Angles of each link.
        :return: Number of collisions (presented as a negative score).
        """

        collision_score = 0

        for n in range(len(self.obstacles)):
            for i in range(len(self.arm_lengths)):
                if self.get_links(thetas)[i].check_wall_collision(self.obstacles[n]):
                    collision_score += 1

        return -collision_score

    def ik_constrained_search(self, target, thetas_guess, max_iters=100):
        raise NotImplementedError

    def plot_robot_state(self, thetas, target=None, filename='robot_arm_state.png'):

        links = self.get_links(thetas)

        for i in range(len(links)):
            plt.scatter(links[i].start[0], links[i].start[1])
            if links[i].check_any_wall_collision(self.obstacles):
                color = "red"
                line_style = "dotted"
            else:
                color = "black"
                line_style = "solid"
            plt.arrow(links[i].start[0], links[i].start[1], links[i].end[0] - links[i].start[0],
                      links[i].end[1] - links[i].start[1], color='{}'.format(color), linestyle='{}'.format(line_style))

        for n in range(len(self.obstacles)):
            plt.axvline(x=self.obstacles[n].loc)

        plt.axis([-sum(self.arm_lengths), sum(self.arm_lengths), 0, sum(self.arm_lengths)])

        if target is not None:
            plt.scatter(target[0], target[1])
            plt.annotate("Target",  # this is the text
                         target,  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

        return plt.savefig('{}'.format(filename))


class Link:

    def __init__(self, start, end):
        """
        Represents a finite line segment in the XY plane, with start and ends given as 2-vectors
        :param start: A length 2 Numpy array
        :param end: A length 2 Numpy array
        """
        self.start = start
        self.end = end

    def __repr__(self):
        return '<Link: ({:.3f}, {:.3f}) to ({:.3f}, {:.3f})>'.format(self.start[0], self.start[1],
                                                                     self.end[0], self.end[1])

    def __str__(self):
        return self.__repr__()

    def check_wall_collision(self, wall):
        if self.start[0] < wall.loc < self.end[0] or self.start[0] > wall.loc > self.end[0]:
            return True
        else:
            return False

    def check_any_wall_collision(self, all_walls):
        """
        Checks to see if a link is hitting any walls.
        :param all_walls: List of all walls
        :return:  Returns True if there is a collision, False if there isn't.
        """
        for n in range(len(all_walls)):
            if self.start[0] < all_walls[n].loc < self.end[0] or self.start[0] > all_walls[n].loc > self.end[0]:
                return True
        return False


class VerticalWall:

    def __init__(self, loc):
        """
        A VerticalWall represents a vertical line in space in the XY plane, of the form x = loc.
        :param loc: A scalar value
        """
        self.loc = loc

    def __repr__(self):
        return '<VerticalWall at x={:.3f}>'.format(self.loc)


if __name__ == '__main__':
    # Example of initializing a 3-link robot arm
    arm = RobotArm(2, 1, 2, obstacles=[VerticalWall(3.2)])
    print(arm)

    # Get the end-effector position of the arm for a given configuration
    thetas = [np.pi / 4, 0, -np.pi / 4]
    pos = arm.get_ee_location(thetas)
    print('End effector is at: ({:.3f}, {:.3f})'.format(*pos))

    # Get each of the links for the robot arm, and print their start and end points
    links = arm.get_links(thetas)

    for i, link in enumerate(links):
        print('Link {}:'.format(i))
        print('\tStart: ({:.3f}, {:.3f})'.format(*link.start))
        print('\tEnd: ({:.3f}, {:.3f})'.format(*link.end))

    dist = []
    num_calls = []
    this_thetas = []
    """
    for i in np.linspace(0, 250, 50, endpoint=False):
        my_target = [-1.5, 1.5]
        guess = ([0, 0, 0])
        output = arm.ik_fmin_search(my_target, guess, max_calls=i)
        this_thetas.append(output[0])
        dist.append(output[1])
        num_calls.append(output[2])
        print(arm.distance_from_target(output[0], my_target))

    arm.plot_robot_state(this_thetas[-1], my_target, filename='Problem_3_Plot')
    plt.scatter(num_calls, dist)
    plt.show()
    """
    arm.plot_robot_state([0.2, 0.4, 0.6])
