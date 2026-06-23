import time
import numpy as np
import threading
import queue
import treadmill

import zmq
import nidaqmx
from nidaqmx.constants import AcquisitionType
import random, string
from itertools import product

from typing import Callable, Any
from skopt import Optimizer
import warnings
warnings.filterwarnings("ignore")

import dearpygui.dearpygui as dpg

##############################################################
# Object defintions 
class Data_Storage_Class():
    """Allocated arrays for storing data and later saving to .npz files"""
    def __init__(self, trial_length_min, num_DATA, num_COMMS, num_EXO):
        self.trial_length_frames = round(trial_length_min*60*75)
        self.DATA = np.zeros((num_DATA, self.trial_length_frames))
        self.COMMS = np.zeros((num_COMMS, self.trial_length_frames))
        self.EXO = np.zeros((num_EXO, self.trial_length_frames))
        self.TREAD = np.zeros((6, self.trial_length_frames))
        self.recording_start_time = time.time()
        self.approx_frame_num = 0
        self.init_last_frame = np.zeros((13,1))
    
    def restart(self, trial_length_min):
        self.trial_length_frames = round(trial_length_min*60*75)
        self.DATA = np.zeros((np.size(self.DATA,0), self.trial_length_frames))
        self.COMMS = np.zeros((np.size(self.COMMS,0), self.trial_length_frames))
        self.EXO = np.zeros((np.size(self.EXO,0), self.trial_length_frames))
        self.TREAD = np.zeros((6, self.trial_length_frames))
        self.recording_start_time = time.time()
        self.init_last_frame = np.zeros((13,1))

    def write_DATA(self, frame_num, values):
        self.DATA[:, frame_num] = values
        self.approx_frame_num = frame_num

    def write_COMMS(self, frame_num, values):
        self.COMMS[:, frame_num] = values

    def write_EXO(self, frame_num, values, cost, time):
        self.EXO[:-2, frame_num] = values
        self.EXO[-2, frame_num] = cost
        self.EXO[-1, frame_num] = time

    def write_TREAD(self, frame_num, values):
        self.TREAD[:, frame_num] = values

    def save(self):
        id = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        np.savez("raw_DATA/"+id+"_RENAME.npz", DATA = self.DATA, COMMS = self.COMMS, EXO = self.EXO, TREAD = self.TREAD, START = self.recording_start_time)

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

class HumanInLoopOpt():
    """Optimizer structure used for HILO"""
    def __init__(self, DeviceState, search_space, grid_density):
        self.DeviceState = DeviceState
        self.search_space = search_space
        self.grid = np.array(np.meshgrid(*[np.linspace(lo, hi, grid_density) for lo, hi in search_space])).reshape(len(search_space), -1).T
        self.optimizer = Optimizer(
            dimensions=self.search_space,
            base_estimator="GP",
            acq_func="EI",
        )

    def restart(self):
        self.optimizer = Optimizer(
            dimensions=self.search_space,
            base_estimator="GP",
            acq_func="EI",
        )

    def step(self):
        x = self.optimizer.ask()
        self.force(x)
        return x

    def force(self, x):
        self.DeviceState.set_btn("Advance Optim.", False)

        log_write("Update to:")
        log_write(", ".join(map(str, x)))
        dpg.set_value("rdt_Exo", ", ".join(map(str, x)))
        log_write(">Press [Advance] to Continue\n")
        self.DeviceState.set_locked("Advance Optim.", False)
        while self.DeviceState.get("Advance Optim.") == False:
            time.sleep(0.1)

        self.DeviceState.set_locked("Advance Optim.", True)
        log_write("Collecting...")
        return x
    
    def observe(self, x, sensor_data):
        score = self.cost(sensor_data, x)
        result = self.optimizer.tell([x], score) 
        return score, result

    def cost(self, sensor_data, x):
        V_L = sensor_data[0]
        V_R = sensor_data[1]
        return abs(V_L - V_R*0.8) #+ sum(x)

##############################################################
# Thread functions
def collect_data(DeviceState):
    """Reads in data from nidaq and passes is to distributor thread (init_treadmill)"""
    try:
        log_write("Opened Thread: RD")
        DeviceState.set_locked("Self Paced Mode", False)
        DeviceState.set_locked("HILO", False)
        stop_event = DeviceState._thread_stop["Recording Data"]

        #####################################

        frame_num = 1
        with nidaqmx.Task() as task:
            # Add an analog input voltage channel (e.g., 'Dev1/ai0')
            config = nidaqmx.constants.TerminalConfiguration(10083)
            channel_names = ["Dev1/ai31", "Dev1/ai23", "Dev1/ai30", "Dev1/ai22", "Dev1/ai29", "Dev1/ai21", #left
                            "Dev1/ai20", "Dev1/ai27", "Dev1/ai19", "Dev1/ai26", "Dev1/ai18", "Dev1/ai25"] #right
                            # Fx           Fy           Fz           Mx           My           Mz
            for channel_name in channel_names:
                task.ai_channels.add_ai_voltage_chan(channel_name,terminal_config=config)

            #Configure timing for continuous acquisition at 1000 Hz
            task.timing.cfg_samp_clk_timing(
                rate=100,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=1
            )

            log_write(">>>Data Collecting")
            DeviceState.DS.restart(DeviceState.LENGTH)
            DeviceState.EVAL.restart()
            DeviceState.OPT.restart()
            task.start()

            while not stop_event.is_set():
                curr_time = time.time()

                frame_raw = task.read(number_of_samples_per_channel=1)
                frame = np.append(np.array(frame_raw), [curr_time-DeviceState.DS.recording_start_time])

                if frame_num < DeviceState.DS.trial_length_frames:
                    DeviceState._queues["Recording Data"].put((frame, frame_num))
                else:
                    log_write("=== Ran Out of Space ===")
                    stop_event.set()

                frame_num += 1
    except Exception as e:
        print("Collection Thread Error:") 
        print(e)
    finally:
        log_write("Closed Thread: RD")
        DeviceState.DS.save()
        log_write(">>>Saved Results")
        thread_names = list(DeviceState._queues.keys())
        thread_names.pop(0)
        for thread_name in thread_names:
            DeviceState.set_btn(thread_name, False)
        DeviceState.set_locked("Self Paced Mode", True)
        DeviceState.set_locked("HILO", True)

def init_treadmill(DeviceState):
    """Makes computations over collected data and sends commands to appropriate secondary threads"""
    try:
        log_write("Opened Thread: IT")
        DeviceState.set_locked("Initiate Treadmill", True)
        DeviceState.set_locked("Recording Data", False)
        stop_event = DeviceState._thread_stop["Initiate Treadmill"]
        
        #####################################

        while not stop_event.is_set():
            read = DeviceState._queues["Recording Data"].get()
            if read is None:
                DeviceState._queues["Recording Data"].task_done()
                continue
            else:
                frame, frame_num = read

            values = np.append(frame[:12], [DeviceState.EVAL.target*0.9, frame[12]])
            DeviceState.DS.write_DATA(frame_num, values)

            message, message_ready = DeviceState.EVAL.haptic_logic(frame, DeviceState.DS.init_last_frame, frame_num)
            bayes_msg, bayes_msg_ready = DeviceState.EVAL.bayesian_logic(frame, DeviceState.DS.init_last_frame, frame_num)
            speed_msg, speed_msg_ready = DeviceState.EVAL.speed_logic(frame, DeviceState.DS.init_last_frame, frame_num)

            DeviceState.DS.init_last_frame = frame

            if message_ready and DeviceState._states["ROS2 Comms"]:
                DeviceState._queues["ROS2 Comms"].put((message, frame_num))
                message_ready = False

            if bayes_msg_ready and DeviceState._states["HILO"] and DeviceState._states["Advance Optim."]:
                DeviceState._queues["HILO"].put((bayes_msg, frame_num))
                bayes_msg_ready = False
            
            if speed_msg_ready and DeviceState._states["Self Paced Mode"]:
                DeviceState._queues["Self Paced Mode"].put((speed_msg, frame_num))
                speed_msg_ready = False

            DeviceState._queues["Recording Data"].task_done()
    except Exception as e: 
        print("IT Thread Error:")
        print(e)
    finally:
        log_write("Closed Thread: IT")
        DeviceState.set_locked("Recording Data", True)

def send_result(DeviceState):
    """Supports local socket based communication to a WSL portal, which can run ROS2 and populate a Node"""
    try:
        log_write("Opened Thread: COM")
        stop_event = DeviceState._thread_stop["ROS2 Comms"]

        #####################################

        print("Connecting to server…")
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")

        try:
            heard = socket.recv()
            print("COMMS: [%s]" % (heard))
        except Exception as e: 
            print("Failed To Connect")
            print(e)
            socket.close()
            context.term()
        
        while not stop_event.is_set():
            read = DeviceState._queues["ROS2 Comms"].get()
            if read is None:
                DeviceState._queues["ROS2 Comms"].task_done()
                break
            else:
                message, frame_num = read

            curr_time = time.time()
            socket.send_string(str(message))
            response = socket.recv()
            log_write("COMMS: [S: %s] [H: %s]" % (message, response))

            values = np.append(message, [curr_time - DeviceState.start_time])
            DeviceState.DS.write_COMMS(frame_num, values)

            values = np.append([0,0], [curr_time - DeviceState.start_time - 0.01])
            DeviceState.DS.write_COMMS(frame_num - 1, values)
            values = np.append([0,0], [curr_time - DeviceState.start_time + 0.01])
            DeviceState.DS.write_COMMS(frame_num + 1, values)

            DeviceState._queues["ROS2 Comms"].task_done()
    except Exception as e: 
        print("COMM Thread Error:")
        print(e)
    finally:
        log_write("Closed Thread: COM")
        socket.close()
        context.term()

def bayesian(DeviceState):
    """Initiates a HILO session, and tells optimizer when to evaluate / advance"""
    try:
        log_write("Opened Thread: OPT")
        stop_event = DeviceState._thread_stop["HILO"]

        #####################################

        result = None
        search = DeviceState.search_method
        x = DeviceState.init_params #INITIAL PARAMETERS OF DEVICE
        continue_from_trials = DeviceState.continue_from
        history = np.zeros((20,2))
        counter = 0

        if continue_from_trials != None:
            for trial in continue_from_trials:
                log_write(">Loading in the following data:")
                DATA = np.load(trial)
                EXO = DATA["EXO"]
                START = DATA["START"]

                EXO[-1,:]=EXO[-1,:]-START
                bad_rows = []
                for i in range(np.shape(EXO)[1]):
                    if EXO[-1,i]<0:
                        bad_rows.append(i)
                EXO = np.delete(EXO, bad_rows, axis=1)

                for i in range(np.size(EXO, 1)):
                    log_write(*EXO[:,i])
                    DeviceState.OPT.optimizer.tell(EXO[0:1,i].tolist(), EXO[1,i]) 

        DeviceState.OPT.force(x)
        step_num = 0
        log_write(">>>Optimizer Starting")

        while not stop_event.is_set():
            read = DeviceState._queues["HILO"].get()
            if read is None:
                DeviceState._queues["HILO"].task_done()
                break
            else:
                message, frame_num = read

            msg_arr = np.array(message).reshape(1,-1)
            history = np.concatenate((history, msg_arr), axis=0)
            history = np.delete(history, [0], axis=0)
            effect = np.mean(history[:10, :], axis=0)
            # ref = abs(history-effect)
            # compare = ref<epsilon*effect

            counter += 1

            #if (compare.all() and counter>=20) or counter>30:
            if counter >= 10:
                cost, result = DeviceState.OPT.observe(x, effect)
                #print(f"{step_num}, S{counter}: [{x[0]}, {x[1]}, {x[2]}] = [{cost}]")
                readout = ", ".join(map(str, x))
                log_write(f"x: [{readout}] = C: [{cost:.3f}]")
                DeviceState.DS.write_EXO(step_num, x, cost, time.time())
                counter = 0

                step_num += 1
                if search == "grid":
                    if step_num >= len(DeviceState.OPT.grid):
                        break

                    x = DeviceState.OPT.grid[step_num]
                    DeviceState.OPT.force(x)
                    
                if search == "bayes":
                    x = DeviceState.OPT.step()

            DeviceState._queues["HILO"].task_done()

    except Exception as e: 
        print("OPT Thread Error:")
        print(e)
    finally:
        log_write("Closed Thread: OPT")
        if result != None:
            #print(f"Optimal: [{result.x[0]}, {result.x[1]}] = [{result.fun:.2f}]")
            log_write(f"Optimal: [{result.x}] = [{result.fun:.2f}]")

def self_paced(DeviceState):
    """Dynamic controller for preferred walking speed. Will behave poorly if treadmill is not zeroed"""
    try:
        log_write("Opened Thread: TR")
        stop_event = DeviceState._thread_stop["Self Paced Mode"]

        #####################################

        #time.sleep(3)
        v_gain = 0.1
        a_gain = 1

        speed = float(dpg.get_value("inpt_INPUT: Velocity"))
        curr_speed = speed
        pos_zero = 0.75
        last_position = 0.75
        last_step_time = 0
        last_delta = 0
        treadmill.set_treadmill(speed*1000,speed*1000,100,100)
        #time.sleep(10)
        log_write(">>>Treadmill Ready")

        while not stop_event.is_set():
            read = DeviceState._queues["Self Paced Mode"].get()
            if read is None:
                DeviceState._queues["Self Paced Mode"].task_done()
                break
            else:
                message, frame_num = read

            delta_t = time.time() - last_step_time
            last_step_time = time.time()

            position_estimate = np.mean(message)
            if position_estimate < 0: position_estimate = pos_zero
            if position_estimate > 2: position_estimate = pos_zero

            v_rel = (position_estimate-last_position)/delta_t
            centering_velocity = position_estimate - pos_zero
            if abs(centering_velocity) < 0.1: centering_velocity = 0

            last_position = position_estimate


            delta = (centering_velocity + v_rel) * v_gain
            acceleration = abs(speed - curr_speed) * a_gain

            #############################

            ### SAFETY LIMITS
            if acceleration >= 0.25: acceleration = 0.25

            jerk = delta - last_delta
            if abs(jerk) >= 0.15: delta = 0
            
            if delta >= 0.15: delta = 0.15
            if delta <= -0.15: delta = -0.15
            ###

            speed = speed + delta
            last_delta = delta

            if speed<0: speed = 0
            if speed>2: speed = 2

            #print(f"P:[{(position_estimate):.2f}], V:[{speed:.2f}], A:[{acceleration:.2f}], CV:[{centering_velocity:.2f}], VR:[{v_rel:.2f}]")

            curr_speed, _, _ = treadmill.set_treadmill((speed)*1000, (speed)*1000, (acceleration)*1000, (acceleration)*1000)
            curr_speed = curr_speed/1000
            DeviceState.DS.write_TREAD(frame_num, [position_estimate, speed, curr_speed, acceleration, centering_velocity, v_rel])
            dpg.set_value("rdt_Velocity", f"V: {curr_speed:.2f} [m/s]" )

            DeviceState._queues["Self Paced Mode"].task_done()

    except Exception as e: 
        print("TR Thread Error:")
        print(e)
    finally:
        log_write("Closed Thread: TR")

#####################################################################################################
# Main class and GUI control 
class DeviceState:
    """High Level Controller that holds button states, opens/closes threads, and commands treadmill"""

    def __init__(self, search_space, grid, trial_length, init_params, search_method, continue_from):
        self.DS = Data_Storage_Class(trial_length, num_DATA=14, num_COMMS=3, num_EXO=4)
        self.EVAL = Evaluator_Class(self.DS)
        self.OPT = HumanInLoopOpt(self, search_space, grid)
        self.START = time.time()
        self.LENGTH = trial_length
        self.init_params = init_params
        self.search_method = search_method
        self.continue_from = continue_from
        self.fixed_vel = 0; self.fixed_acc = 0

        self._states = {
            "Initiate Treadmill": False, 
            "Recording Data": False, 
            "Self Paced Mode": False, 
            "HILO": False, 
            "ROS2 Comms": False,
            "Advance Optim.": False,
            "SET": False,
            "STOP": False,
            }
        self._locked = self._states.copy()
        self._thread_stop = {
            "Initiate Treadmill": threading.Event(), 
            "Recording Data": threading.Event(), 
            "Self Paced Mode": threading.Event(), 
            "HILO": threading.Event(), 
            "ROS2 Comms": threading.Event(),
            }
        self._queues = {
            "Initiate Treadmill": queue.Queue(), 
            "Recording Data": queue.Queue(), 
            "Self Paced Mode": queue.Queue(), 
            "HILO": queue.Queue(), 
            "ROS2 Comms": queue.Queue(),
            }
        # self._threads = {
        #     "Initiate Treadmill": init_treadmill(self), 
        #     "Recording Data": collect_data(self), 
        #     "Self Paced Mode": self_paced(self), 
        #     "HILO": bayesian(self), 
        #     "ROS2 Comms": send_result(self),
        #     }
        self._threads = {
            "Initiate Treadmill": init_treadmill, 
            "Recording Data": collect_data, 
            "Self Paced Mode": self_paced, 
            "HILO": bayesian, 
            "ROS2 Comms": send_result,
            }
        

    def set_locked(self, name, locked: bool):
        self._locked[name] = locked
        dpg.configure_item(f"btn_{name}", enabled=not locked)

    def is_locked(self, name) -> bool:
        return self._locked[name]

    def set_btn(self, name, value: bool):
        self._states[name] = value
        _refresh_button(name, value)
        if name in list(self._threads.keys()):
            if value == True:
                with self._queues[name].mutex:
                    self._queues[name].queue.clear()
                    self._queues[name].all_tasks_done.notify_all()
                    self._queues[name].unfinished_tasks = 0
                self._thread_stop[name].clear()
                threading.Thread(target=self._threads[name], args=[self]).start()
            if value == False:
                self._thread_stop[name].set()
                self._queues[name].put(None)
        if name == "STOP" and value == True:
            self.big_red_stop()
        if name == "SET" and value == True:
            self.fixed_vel = float(dpg.get_value("inpt_INPUT: Velocity"))
            self.fixed_acc = float(dpg.get_value("inpt_INPUT: Acceleration"))
            self.set_speed_fixed()

    def get(self, name) -> bool:
        return self._states[name]

    def toggle(self, name):
        if self._locked[name]:
            return
        new_val = not self._states[name]
        self.set_btn(name, new_val)

    def set_speed_fixed(self):
        self.set_btn("Self Paced Mode", False)
        treadmill.set_treadmill(self.fixed_vel*1000,self.fixed_vel*1000,self.fixed_acc*1000,self.fixed_acc*1000)
        self.DS.write_TREAD(self.DS.approx_frame_num, [0, self.fixed_vel, 0, self.fixed_acc, 0, 0])
        dpg.set_value("rdt_Velocity", f"V: {self.fixed_vel:.2f} [m/s]" )
        self.set_btn("STOP", False)
        self.set_btn("SET", False)

    def big_red_stop(self):
        self.fixed_vel = 0
        self.fixed_acc = 0.25
        dpg.set_value("inpt_INPUT: Velocity", f"{0:.2f}" )
        dpg.set_value("inpt_INPUT: Acceleration", f"{0.25:.2f}" )
        self.set_speed_fixed()

# --- Helpers ---

def _refresh_button(name: str, value: bool):
    """Update button label and color"""
    label = f"{name}: {'ON' if value else 'OFF'}"
    dpg.set_item_label(f"btn_{name}", label)
    theme_tag = "theme_on" if value else "theme_off"
    dpg.bind_item_theme(f"btn_{name}", theme_tag)

def log_write(message: str):
    """Append a timestamped line to the log box"""
    current = dpg.get_value("log_box")
    timestamp = time.strftime("%H:%M:%S")
    dpg.set_value("log_box", current + f"[{timestamp}]  {message}\n")
    # Scroll to bottom
    dpg.set_y_scroll("log_panel", dpg.get_y_scroll_max("log_panel"))

def mini_thread(DeviceState):
    """for testing // all threads should follow this format"""
    try:
        log_write("Opened Thread: MINI")
        stop_event = DeviceState._thread_stop["Recording Data"]

        ##############################

        while not stop_event.is_set():
            time.sleep(1)
            log_write("test")
    except Exception as e: 
        print("MINI Thread Error:")
        print(e)
    finally:
        log_write("Closed Thread: MINI")

# --- GUI setup ---

def build_gui(state: DeviceState):
    """Purely visual components of the GUI. If an element is added here, it should also be added to DeviceState"""
    dpg.create_context()
    dpg.create_viewport(title="Device Control", width=500, height=850, resizable=True)

    # --- Themes for ON / OFF button states ---
    with dpg.theme(tag="theme_on"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (80, 180, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (100, 200, 100))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (60, 160, 60))

    with dpg.theme(tag="theme_off"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (60, 60, 70))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (80, 80, 90))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (40, 40, 50))

    with dpg.window(label="Device Control", tag="main_window", no_close=True,
                    no_move=True, no_resize=True, no_title_bar=True):

        dpg.add_spacer(height=4)

        # --- Buttons ---
        for name in ["Initiate Treadmill", "Recording Data", "Self Paced Mode", "HILO", "ROS2 Comms", "Advance Optim.", "SET", "STOP"]:
            if name == "Advance Optim.":
                dpg.add_separator()
            dpg.add_button(
                label=f"{name}: OFF",
                tag=f"btn_{name}",
                width=260,
                height=36,
                callback=lambda _, __, n: state.toggle(n),
                user_data=name,
                enabled=False,   # Starts locked
            )
            dpg.bind_item_theme(f"btn_{name}", "theme_off")
            dpg.add_spacer(height=4)

        dpg.add_separator()
        dpg.add_spacer(height=6)

        # --- Inputs field ---
        for name in ["INPUT: Velocity", "INPUT: Acceleration"]:
            dpg.add_text(f"{name}:")
            dpg.add_input_text(tag=f"inpt_{name}", default_value="0",
                           width=260, readonly=False)
            dpg.add_spacer(height=4)

        dpg.add_separator()
        dpg.add_spacer(height=6)

        # --- Status field ---
        for name in ["Time", "Velocity", "Exo"]:
            dpg.add_text(f"{name}:")
            dpg.add_input_text(tag=f"rdt_{name}", default_value="...",
                           width=260, readonly=True)
            dpg.add_spacer(height=4)

        dpg.add_separator()
        dpg.add_spacer(height=6)

        # --- Log box ---
        dpg.add_text("Log:")
        with dpg.child_window(tag="log_panel", width=470, height=160, horizontal_scrollbar=False):
            dpg.add_input_text(
                tag="log_box",
                default_value="",
                width=450,
                height=140,
                multiline=True,
                readonly=True,
                tab_input=False,
            )

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    