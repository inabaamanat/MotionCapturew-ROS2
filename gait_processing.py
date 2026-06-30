import time
import numpy as np

import rclpy
from rclpy.node import Node

# Replace Data_Storage with ROS topic messages or rosbag logging. need to create another package for that. For now, just use a list to store data.
class Evaluator_Class():
    """Logical core of program, holds multiple functions used to evaluate changes in collected data"""
    def __init__(self, Data_Storage):
        self.DS = Data_Storage
        self.threshold = 0.1
        self.window_start_R = 0; self.window_start_L = 0
        self.window_end_R = 0; self.window_end_L = 0
        self.observation_window = []; self.num_strides = 0
        self.wiggle = 0.9; self.target = 1
        self.primed_L = False; self.primed_R = False
        self.time_of_last_msg = 0
        self.interval = 0.1
        self.l_foot_pos = 0
        self.r_foot_pos = 0
        self.treadmill_ready = False
    
    def restart(self):
        self.threshold = 0.1
        self.window_start_R = 0; self.window_start_L = 0
        self.window_end_R = 0; self.window_end_L = 0
        self.observation_window = []; self.num_strides = 0
        self.wiggle = 0.9; self.target = 1
        self.primed_L = False; self.primed_R = False
        self.time_of_last_msg = 0
        self.interval = 0.1
        self.l_foot_pos = 0
        self.r_foot_pos = 0
        self.treadmill_ready = False

    def check_for_transition(self, frame, last_frame):
        if (frame[2] < self.threshold) and (last_frame[2] >= self.threshold):
            state = "falling edge left"
            self.num_strides += 1
        elif (frame[8] < self.threshold) and (last_frame[8] >= self.threshold):
            state = "falling edge right"
            self.num_strides += 1
        elif (frame[2] > self.threshold) and (last_frame[2] <= self.threshold):
            state = "leading edge left"
        elif (frame[8] > self.threshold) and (last_frame[8] <= self.threshold):
            state = "leading edge right"
        else:
            state = "none"
        
        return state

    def update_target(self, window, side):
        if side == "L":
            last_max = max(window[2,:])
        if side == "R":
            last_max = max(window[8,:])

        if self.num_strides < 10:
            self.observation_window.append(last_max)
        else:
            self.observation_window.pop(0)
            self.observation_window.append(last_max)

        self.target = np.mean(self.observation_window)

    def update_target_simple(self, frame, frame_num):
        last = frame[2] + frame[8] 

        if frame_num < 250:
            self.observation_window.append(last)
        else:
            self.observation_window.pop(0)
            self.observation_window.append(last)

        self.target = np.mean(self.observation_window)
        

    def sweet_spot(self, frame, last_frame, frame_num):
        state = self.check_for_transition(frame, last_frame)
        message_ready = False
        message = [0,0]
        
        if state == "falling edge right":
            self.window_end_R = frame_num
            window_R = self.DS.DATA[:,self.window_start_R:self.window_end_R]
            self.update_target(window_R, "R")
            self.window_start_R = self.window_end_R

        if state == "falling edge left":
            self.window_end_L = frame_num
            window_L = self.DS.DATA[:,self.window_start_L:self.window_end_L]
            self.update_target(window_L, "L")
            self.window_start_L = self.window_end_L

        if state == "leading edge left":
            self.primed_L = True
        if self.primed_L == True and frame[2]>=self.target*self.wiggle:
            message_ready = True
            message = [255,0]
            self.primed_L = False

        if state == "leading edge right":
            self.primed_R = True
        if self.primed_R == True and frame[8]>=self.target*self.wiggle:
            message_ready = True
            message = [0,255]
            self.primed_R = False

        return message, message_ready
    
    def symmetry(self, frame, last_frame, frame_num):
        state = self.check_for_transition(frame, last_frame)
        message_ready = False
        message = [0,0]

        if state == "leading edge right":
            self.window_end_R = frame_num
            window = self.DS.DATA[:,self.window_start_R:self.window_end_R]

            message_ready = True
            self.window_start_R = self.window_end_R

        if state == "leading edge left":
            self.window_end_L = frame_num
            window = self.DS.DATA[:,self.window_start_L:self.window_end_L]

            message_ready = True
            self.window_start_L = self.window_end_L

        if message_ready:
            L_max = max(window[2,:])
            R_max = max(window[8,:])
            message = [L_max, R_max]

        return message, message_ready
    
    def continuous_feedback(self, frame, last_frame, frame_num):
        message_ready = False; message = [0,0]
        self.update_target_simple(self, frame, frame_num)
        
        if self.time_of_last_msg - time.time() > self.interval:
            message_ready = True
            message = [min(255 * frame[2]/self.target, 255), min(255 * frame[8]/self.target, 255)]

            self.time_of_last_msg = time.time()
        
        return message, message_ready
    
    def position_estimate(self, frame, last_frame, frame_num):
        state = self.check_for_transition(frame, last_frame)
        message_ready = False
        message = [0,0]

        if state == "leading edge right":
            self.r_foot_pos = frame[3]/frame[2]
            message[1] = self.r_foot_pos
            message_ready = True
        if state == "leading edge left":
            self.l_foot_pos = frame[9]/frame[8]
            message[0] = self.l_foot_pos
            message_ready = True

        if state == "falling edge right": 
            self.r_foot_pos = frame[3]/frame[2]
            message[1] = self.r_foot_pos
        if state == "falling edge left": 
            self.l_foot_pos = frame[9]/frame[8]
            message[0] = self.l_foot_pos

        if message_ready:
            message = [self.l_foot_pos, self.r_foot_pos]

        return message, message_ready
    
    haptic_logic = sweet_spot
    bayesian_logic = symmetry
    speed_logic = position_estimate