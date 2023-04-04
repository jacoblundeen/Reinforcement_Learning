"""
605.649 Introduction to Machine Learning
Dr. Donnelly
Programming Project #05
20221211
Jacob M. Lundeen

For the fifth and final programming assignment, you will implement a reinforcement learner and apply it
to the racetrack problem. The racetrack problem is a popular control problem, often used in reinforcement
learning assignments. The goal is to control the movement of a race car along a pre-defined racetrack. You
want your race car to get from the starting line to the finish line in a minimum amount of time. In this
version of the problem, your agent will be the only racer on the track, so it is more like a time trial than a
full competitive race.
"""

import pandas as pd
import numpy as np
from collections import Counter
import math
import time
from numpy import log2 as log
import pprint
import warnings
import random
from xlwt import Workbook
import os
import time
from copy import deepcopy
from random import shuffle
import winsound

algo_name = "SARSA_Learning"
file_name = "O-track.txt"
track_name = "O_track"
# Variables to indicate sections of track
start = 'S'
goal = 'F'
wall = '#'
track = '.'
max_vel = 5
min_vel = -5

theta = 0.001  # Determine when Q-values stabilize
prob_fail = 0.20  # Probability car will try to take action
prob_success = 1 - prob_fail
train_len = 10
num_races = 10  # How many times the race car does a single time trial from starting position to the finish line
frame = 0.3  # How many seconds between frames printed to the console
max_steps = 500  # Maximum number of steps the car can take during time trial
max_train_iter = 2  # Maximum number of training iterations
vel_range = [*range(min_vel, max_vel + 1)]  # Range of the velocity of the race car in both y and x directions
actions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


# Read in the racetrack
def read_track(filename):
    file = open(filename, 'r')
    Lines = file.readlines()
    racetrack = []
    for i, line in enumerate(Lines):
        if i == 0:
            X = int(line.split(',')[0])
            Y = int(line.split(',')[1])
        elif line == "":
            continue
        else:
            racetrack.append([x for x in line.strip()])
    file.close()
    return racetrack


# Prints track and current position of the racecar
def print_track(racetrack, pos=[0, 0]):
    temp = racetrack[pos[0]][pos[1]]
    racetrack[pos[0]][pos[1]] = "X"
    time.sleep(frame)
    clear()
    for line in racetrack:
        text = ""
        for cha in line:
            text += cha
        print(text)
    racetrack[pos[0]][pos[1]] = temp


# clears print output
def clear():
    os.system('cls')


# Randomize starting position
def start_pos(racetrack):
    pos = []
    for x, row in enumerate(racetrack):
        for y, col in enumerate(row):
            if col == start:
                pos += [(x, y)]
    shuffle(pos)
    return pos[0]


# Calculate new velocity and make sure it is within the range of velocities
def new_vel(old_vel, accel):
    new_x = old_vel[1] + accel[1]
    new_y = old_vel[0] + accel[0]
    # If new velocities are beyond the min and max velocities, set them to their respective min or max
    if new_x < min_vel:
        new_x = min_vel
    if new_x > max_vel:
        new_x = max_vel
    if new_y < min_vel:
        new_y = min_vel
    if new_y > max_vel:
        new_y = max_vel
    return new_y, new_x


# Return the new position based on previous location and new velocities
def new_pos(old_loc, vel):
    return old_loc[0] + vel[0], old_loc[1] + vel[1]


# Find the nearest open cell. Used with crashing. If there is a velocity, search in opposite direction to prevent
# jumping the wall.
def open_cell(racetrack, y_crash, x_crash, vy=0, vx=0, open=[track, start, goal]):
    rows = len(racetrack)
    cols = len(racetrack[0])
    max_rad = max(rows, cols)  # Set coverage for search

    for rad in range(max_rad):
        if vy == 0:
            y_off_range = range(-rad, rad + 1)
        elif vy < 0:
            y_off_range = range(0, rad + 1)
        else:
            y_off_range = range(-rad, 1)
        for y_offset in y_off_range:
            y = y_crash + y_offset
            x_radius = rad - abs(y_offset)
            if vx == 0:
                x_range = range(x_crash - x_radius, x_crash + x_radius + 1)
            elif vx < 0:
                x_range = range(x_crash, x_crash + x_radius + 1)
            else:
                x_range = range(x_crash - x_radius, x_crash + 1)
            for x in x_range:
                if y < 0 or y >= rows:
                    continue
                if x < 0 or x >= cols:
                    continue
                if racetrack[y][x] in open:  # If open cell is found, return it. Else return nothing
                    return y, x
    return


# Calculate new state
def new_state(old_y, old_x, old_vy, old_vx, accel, racetrack, deterministic=False, crash=False):
    if not deterministic:
        if random.random() > prob_success:
            accel = (0, 0)
    new_vy, new_vx = new_vel((old_vy, old_vx), accel)  # New velocity
    temp_y, temp_x = new_pos((old_y, old_x), (new_vy, new_vx))  # New position
    new_y, new_x = open_cell(racetrack, temp_y, temp_x, new_vy, new_vx)  # Nearest open cell
    if new_y != temp_y or new_x != temp_x:  # Handle crashes
        if crash and racetrack[new_y][new_x] != goal:  # Crash scenario 2, reset to starting position
            new_y, new_x = start_pos(racetrack)
        new_vy, new_vx = 0, 0  # Crash scenario one, don't move the car and set velocity to 0
    return new_y, new_x, new_vy, new_vx


# # SARSA Policy
# def value_policy(cols, rows, vel_range, Q, actions):
#     pie = {}
#     for x in range(rows):
#         for y in range(cols):
#             for vx in vel_range:
#                 for vy in vel_range:
#                     pie[(x, y, vx, vy)] = actions[np.argmax(Q[x][y][vx][vy])]  # Find best a for each s
#     return pie


# Value Iteration
def value_iteration(racetrack, eta, gamma, num_train_iter, crash=False, reward=0.0):
    rows = len(racetrack)
    cols = len(racetrack[1])
    # Create V(s)
    values = [[[[random.random() for _ in vel_range] for _ in vel_range] for _ in line] for line in racetrack]

    # Set the finish line states to 0
    for x in range(rows):
        for y in range(cols):
            if racetrack[x][y] == goal:
                for vy in vel_range:
                    for vx in vel_range:
                        values[x][y][vx][vy] = reward

    Q = [[[[[random.random() for _ in actions] for _ in vel_range] for _ in (vel_range)] for _ in line]
         for line in racetrack]  # Initialize Q(s,a)

    # Set finish line to 0
    for x in range(rows):
        for y in range(cols):
            if racetrack[x][y] == goal:
                for vx in vel_range:
                    for vy in vel_range:
                        for i, a in enumerate(actions):
                            Q[x][y][vx][vy][i] = reward

    # Train car
    for t in range(num_train_iter):
        values_prev = deepcopy(values)
        # delta = 0.0
        # For all the possible states s in S
        for x in range(rows):
            for y in range(cols):
                for vx in vel_range:
                    for vy in vel_range:
                        if racetrack[x][y] == wall:
                            values[x][y][vx][vy] = -9.9
                            continue
                        # For each action a in the set of possible actions A
                        for i, a in enumerate(actions):
                            if racetrack[x][y] == goal:
                                r = reward
                            else:
                                r = -1
                            new_x, new_y, new_vx, new_vy = new_state(x, y, vx, vy, a, racetrack, deterministic=True,
                                                                     crash=crash)
                            val_new = values_prev[new_x][new_y][new_vx][new_vy]
                            new_x, new_y, new_vx, new_vy = new_state(x, y, vx, vy, (0, 0), racetrack,
                                                                     deterministic=True,
                                                                     crash=crash)  # New state when taking action (0,0)
                            val_new_fail = values_prev[new_x][new_y][new_vx][
                                new_vy]  # Value of new state from action (0,0)
                            expected_value = (prob_success * val_new) + (prob_fail * val_new_fail)
                            Q[x][y][vx][vy][i] = r + (gamma * expected_value)  # Update Q
                        argMaxQ = np.argmax(Q[x][y][vx][vy])  # Get the action with the highest Q value
                        values[x][y][vx][vy] = Q[x][y][vx][vy][argMaxQ]  # Update values

        # Reset rewards to 0
        for x in range(rows):
            for y in range(cols):
                if racetrack[x][y] == goal:
                    for vx in vel_range:
                        for vy in vel_range:
                            values[x][y][vx][vy] = reward

        # Check to see if we've stabilized
        delta = max([max([max([max([abs(values[x][y][vx][vy] - values_prev[x][y][vx][vy]) for vy in vel_range])
                               for vx in vel_range]) for y in range(cols)]) for x in range(rows)])
        if delta < theta:
            return policy(cols, rows, Q)

    return policy(cols, rows, Q)


# Run race car around track
def run_race(racetrack, policy, crash=False, animate=True):
    track_copy = deepcopy(racetrack)
    starting_pos = start_pos(racetrack)
    y, x = starting_pos
    vy, vx = 0, 0
    stop_clock = 0

    # Race around track
    for i in range(max_steps):
        if animate:
            print_track(track_copy, pos=[y, x])
        a = policy[(y, x, vy, vx)]  # Get the best action given the current state
        if racetrack[y][x] == goal:
            return i
        y, x, vy, vx = new_state(y, x, vy, vx, a, racetrack, crash=crash)
        if vy == 0 and vx == 0:
            stop_clock += 1
        else:
            stop_clock = 0
        if stop_clock == 5:
            return max_steps
    return max_steps


# Main driver of program
def main(alpha, epsilon, eta, gamma, num_train_iter, crash_scenario, no_training_iter):
    print("\nThe race car is training. Please wait...")
    racetrack_name = file_name
    racetrack = read_track(racetrack_name)

    while no_training_iter < max_train_iter:
        total_steps = 0
        if crash_scenario == 1:
            crash = False
        else:
            crash = True

        # Retrieve the policy
        if algo_name == 'Value_Iteration':
            policy = value_iteration(racetrack, eta, gamma, num_train_iter, crash)
        elif algo_name == 'Q_Learning':
            policy = q_learning(racetrack, eta, gamma, num_train_iter, crash)
        elif algo_name == "SARSA_Learning":
            policy = sarsa_learning(racetrack, eta, gamma, num_train_iter, crash)

        for each_race in range(num_races):
            total_steps += run_race(racetrack, policy, crash=crash, animate=False)

        print("Number of training iterations: " + str(no_training_iter))
        if crash_scenario == 1:
            print("Crash Scenario 1, the car is placed in the closest open cell.")
        else:
            print("Crash Scenario 2, the car is returned to its original starting point.")
        print("Average number of steps the racecar needs to take before finding the finish line: " +
              str(total_steps / num_races) + " steps\n")
        print("\nThe race car is training. Please wait...")

        if algo_name == 'Value_Iteration':
            no_training_iter += 1
        elif algo_name == 'Q_Learning' or algo_name == 'SARSA_Learning':
            if no_training_iter == 1:
                no_training_iter += 1
            else:
                no_training_iter += 1
    outfile_tr = open('trace_file.txt', "w")
    outfile_tr.write(str(policy))
    outfile_tr.close()
    return total_steps / num_races


# Return policy pie(s) for state s and action a
def policy(cols, rows, Q):
    pie = {}
    # loop thru every state
    for x in range(rows):
        for y in range(cols):
            for vx in vel_range:
                for vy in vel_range:
                    pie[(x, y, vx, vy)] = actions[np.argmax(Q[x][y][vx][vy])]
    return pie


# Q-Learning algorithm
def q_learning(racetrack, eta, gamma, num_train_iter, crash=False, reward=0.0):
    rows = len(racetrack)
    cols = len(racetrack[0])

    # Initialize Q
    Q = [[[[[random.random() for _ in actions] for _ in vel_range] for _ in (vel_range)] for _ in line] for line in
         racetrack]

    # Set finish line pairs to 0
    for x in range(rows):
        for y in range(cols):
            if racetrack[x][y] == goal:
                for vx in vel_range:
                    for vy in vel_range:
                        for i, a in enumerate(actions):
                            Q[x][y][vx][vy][i] = reward

    for i in range(num_train_iter):
        for x in range(rows):
            for y in range(cols):
                if racetrack[x][y] == goal:
                    Q[x][y] = [[[reward for _ in actions] for _ in vel_range] for _ in vel_range]

        # Select random state
        x = np.random.choice(range(rows))
        y = np.random.choice(range(cols))
        vy = np.random.choice(vel_range)
        vx = np.random.choice(vel_range)
        for t in range(train_len):
            if racetrack[x][y] == goal:
                break
            if racetrack[x][y] == wall:
                break
            a = np.argmax(Q[x][y][vx][vy])  # Choose best action
            # Act and then observe a new state s'
            new_x, new_y, new_vx, new_vy = new_state(x, y, vx, vy, actions[a], racetrack, crash=crash)
            r = -1
            Q[x][y][vx][vy][a] = ((1 - eta) * Q[x][y][vx][vy][a] + eta *
                                  (r + gamma * max(Q[new_x][new_y][new_vx][new_vy])))  # Update Q
            x, y, vx, vy = new_x, new_y, new_vx, new_vy

    return policy(cols, rows, Q)


# SARSA Learning algorithm
def sarsa_learning(racetrack, eta, gamma, num_train_iter, crash=False, reward=0.0):
    rows = len(racetrack)
    cols = len(racetrack[0])

    # Initialize Q
    Q = [[[[[random.random() for _ in actions] for _ in vel_range] for _ in (vel_range)] for _ in line] for line in
         racetrack]

    # Set finish line pairs to 0
    for x in range(rows):
        for y in range(cols):
            if racetrack[x][y] == goal:
                for vx in vel_range:
                    for vy in vel_range:
                        for i, a in enumerate(actions):
                            Q[x][y][vx][vy][i] = reward

    for i in range(num_train_iter):
        for x in range(rows):
            for y in range(cols):
                if racetrack[x][y] == goal:
                    Q[x][y] = [[[reward for _ in actions] for _ in vel_range] for _ in vel_range]

        # Select random state
        x = np.random.choice(range(rows))
        y = np.random.choice(range(cols))
        vy = np.random.choice(vel_range)
        vx = np.random.choice(vel_range)
        for t in range(train_len):
            if racetrack[x][y] == goal:
                break
            if racetrack[x][y] == wall:
                break
            a = np.argmax(Q[x][y][vx][vy])  # Choose best action
            # Act and then observe a new state s'
            new_x, new_y, new_vx, new_vy = new_state(x, y, vx, vy, actions[a], racetrack, crash=crash)
            r = -1
            Q[x][y][vx][vy][a] = Q[x][y][vx][vy][a] + eta * (r + gamma * Q[new_x][new_y][new_vx][new_vy][a] -
                                                             Q[x][y][vx][vy][a])  # Update Q
            x, y, vx, vy = new_x, new_y, new_vx, new_vy

    return policy(cols, rows, Q)


# Main function.
if __name__ == '__main__':
    random.seed(69)
    wb = Workbook()

    # alpha = np.linspace(0, 1, 5)
    # epsilon = np.linspace(0, 1, 5)
    # eta = np.linspace(0, 1, 5)
    # gamma = np.linspace(0, 1, 5)
    et = 0.5
    g = 0.75
    num_train_iter = [5] # np.arange(10000, 1000000, 100000)
    # num_train_iter = [10000, 20000]
    print("1. Starts from the nearest open cell on the track to the place where it crashed.")
    print("2. Returns back to the original starting position.\n")
    crash_scenario = int(input("Please choose crash scenario (1 or 2): "))
    no_training_iter = int(input("Enter the initial number of training iterations: "))

    for iter in num_train_iter:
        count = 1
        print('This is the ' + str(iter) + ' run.')
        sheet1 = wb.add_sheet(str(iter))
        sheet1.write(0, 1, "Eta")
        sheet1.write(0, 2, 'Gamma')
        sheet1.write(0, 3, 'Num_Steps')
        # for g in gamma:
        #     for et in eta:
                # print("Alpha: ", str(a))
                # print("Epsilon: ", str(ep))
        print("Eta: ", str(et))
        print("Gamma: ", str(g))
        steps = main(0.8, 0.9, et, g, iter, crash_scenario, no_training_iter)
        sheet1.write(count, 1, str(et))
        sheet1.write(count, 2, str(g))
        sheet1.write(count, 3, str(steps))
        count += 1

    filename = algo_name + '_' + track_name + '_' + 'Crash_Scenario' + '_' + str(crash_scenario)
    wb.save('%s.xls' % filename)
    winsound.Beep(440, 1000)
