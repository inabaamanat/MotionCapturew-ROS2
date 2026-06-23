import functions_HILO as GUI
import numpy as np

##############################################################
# READ ME:
#   To run this code you must:
#   0) have admin privledges on your device
#   1) have all necessary packages* installed in python enviroment of choice
#       *packages are listed at the top of functions_HILO.py
#       this code was written in 3.11, but may function in newer versions
#   2) install NI-DAQmx software
#   3) be plugged into treadmill connection (grey usb) and daq (black usb)
#   4) install and open Treadmill.exe (found in Teams folder)
#   5) Settings>Remote Communication>Apply>Close Settings>Enable Remote Communication
#   6) run cmd with admin privledges
#   7) create a folder raw_DATA for saved files to be put in
#   8) call: python _main__treadmill.py
#
# Parameters:
#   General:
#       trial_length -> max allocated length of trail in minutes. (it is safe to end early, so good to over estimate)
#   HILO:
#       search_space -> (min, max) of each variable
#       grid_density -> increments per variable of grid search
#       init_params -> initial configuration of device (must be within search_space)
#       search_method -> method for picking next iteration of parameters (supported: "bayes", "grid")
#       continue_from -> list of files containing prior HILO evaluations (of form: ["file1.npz", "file2.npz"])

search_space = [
    (15, 120),
    (15, 120),
]
grid_density = 5
init_params = [15,15]
search_method = "bayes"
continue_from = None

trial_length = 5

##############################################################

GUI_STATES = GUI.DeviceState(search_space, grid_density, trial_length, init_params, search_method, continue_from)

GUI.build_gui(GUI_STATES)
GUI.log_write("Program started.")
# unlock initial buttons
for name in ["Initiate Treadmill", "SET", "STOP"]:
    GUI_STATES.set_locked(name, False)

# main loop
GUI.dpg.start_dearpygui()
GUI.dpg.destroy_context()