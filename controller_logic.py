import time
import numpy as np

#import rclpy
#from rclpy.node import Node

#from treadmill_driver import treadmill_api
#from treadmill_interfaces.msg import TreadmillCommand


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