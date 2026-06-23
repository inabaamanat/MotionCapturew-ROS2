clear
pause(5)

%%%%
% I reccomend you run this code once before collecting an actual trial, as
% the parallel pool needs to warm up (else the first trial gets truncated).
% 
% Order of Events:
% 1. Turn on treadmill (Big red lever on grey box)
% 2. Click white "Start" button on the remote (small grey box on railing)
% 3. Open Treadmill App
% 4. Open settings in app (bottom right)
% 5. Click box for "Remote Control" (bottom section)
% 6. Click "Okay" on popup
% 7. Click "Apply" then "Ok" to close settings
% 8. Click "Enable Remote Control" (bottom right)
% 9. Navigate back to matlab
% 10. Right click remoteTreadmill folder and "Add to Path"
% 11. In command line type "STOP" -- This will run "STOP.m" and force the
% parallel pool to start
% 12. Test treadmill connection by typing 
% "setTreadmill(0.1*1000,0.1*1000)" - Treadmill should start
% 13. Try stopping the treadmill by typing "STOP"
% 14. Set desired recording time and speed (highlighted below)
% 15. Click "Run" !
% 16. SAVE YOUR DATA -- it should auto-save to a file "RENAME_lastTrial.m",
% be sure to rename this file else it will be overwritten.


starttime = datetime("now","Format","HH:mm:ss.SSS");
disp("start: " + string(starttime))

parfeval(@treadmill_commands,0);
f = parfeval(@collect_data,1);
DATA = fetchOutputs(f);

plot(DATA.Time, DATA.Variables)
xlabel("Time (s)")
ylabel("Amplitude")
legend(DATA.Properties.VariableNames, "Interpreter", "none")

save("RENAME_lastTrial", "DATA")

function [] = treadmill_commands()
    % For the TF trials, we run at 1 speed so it is much more simple.
    % Change the speed with col. 1 (left belt) and 2 (right belt)
    setpoints = [
        1.1 1.1 0.05 0.05
    ];
    pause(5)
    setTreadmill(setpoints(1,1)*1000,setpoints(1,2)*1000,setpoints(1,3)*1000,setpoints(1,4)*1000)
    pause(60*5) %This value should be the length of the trial in seconds
    
    setTreadmill(0, 0,0.05*1000,0.05*1000) %stops treadmill at end, do not remove.
end


function [data_f] = collect_data()
    d = daq("ni");
    
    ch1 = addinput(d,"Dev1","ai31","Voltage"); ch1.TerminalConfig = "SingleEnded";
    ch2 = addinput(d,"Dev1","ai23","Voltage"); ch2.TerminalConfig = "SingleEnded";
    ch3 = addinput(d,"Dev1","ai30","Voltage"); ch3.TerminalConfig = "SingleEnded";
    ch4 = addinput(d,"Dev1","ai22","Voltage"); ch4.TerminalConfig = "SingleEnded";
    ch5 = addinput(d,"Dev1","ai29","Voltage"); ch5.TerminalConfig = "SingleEnded";
    ch6 = addinput(d,"Dev1","ai21","Voltage"); ch6.TerminalConfig = "SingleEnded";
    ch7 = addinput(d,"Dev1","ai28","Voltage"); ch7.TerminalConfig = "SingleEnded"; 
    ch8 = addinput(d,"Dev1","ai20","Voltage"); ch8.TerminalConfig = "SingleEnded";
    ch9 = addinput(d,"Dev1","ai27","Voltage"); ch9.TerminalConfig = "SingleEnded";
    ch10 = addinput(d,"Dev1","ai19","Voltage"); ch10.TerminalConfig = "SingleEnded";
    ch11 = addinput(d,"Dev1","ai26","Voltage"); ch11.TerminalConfig = "SingleEnded";
    ch12 = addinput(d,"Dev1","ai18","Voltage"); ch12.TerminalConfig = "SingleEnded";
    ch13 = addinput(d,"Dev1","ai25","Voltage"); ch13.TerminalConfig = "SingleEnded";
    ch14 = addinput(d,"Dev1","ai17","Voltage"); ch14.TerminalConfig = "SingleEnded";
    
    d.Rate = 75; %recording rate of the treadmill.. I keep at 75 Hz cause that matches Xsensor. Not sure what maximum is.
    
    [data_f,starttime] = read(d,seconds((5*60)+10)); %seconds(...) is the length that it records data for. This should match the total time from the above function
    data_f.Time = data_f.Time + starttime;
    
    allVars = 1:width(data_f);
    newNames = ["FxL" "FyL" "FzL" "MxL" "MyL" "MzL" "ZL" "FxR" "FyR" "FzR" "MxR" "MyR" "MzR" "ZR"]; %w.r.t Y vertical, X a/p, and Z m/l 
    data_f = renamevars(data_f,allVars,newNames);
end