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
