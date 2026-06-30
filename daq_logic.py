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
