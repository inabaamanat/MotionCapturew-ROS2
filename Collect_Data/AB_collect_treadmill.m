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

function [] = treadmill_commands()
    %Define your desired setpoints in the below array. (speed (col. 1 - left belt, 2 - right belt)
    %and acc. (col. 3, 4)). The default setup is 15 different speeds spaced
    %out by a minuite. You can change the time between changes below
    setpoints = [
        0.5 0.5 0.1 0.1
        0.75 0.75 0.1 0.1
        1.0 1.0 0.1 0.1
        1.2 1.2 0.1 0.1
        1.1 1.1 0.2 0.2
        0.9 0.9 0.2 0.2
        0.7 0.7 0.2 0.2
        0.5 0.5 0.2 0.2
        0.6 0.6 0.3 0.3
        0.8 0.8 0.3 0.3
        1.0 1.0 0.3 0.3
        1.2 1.2 0.3 0.3
        1.4 1.4 0.1 0.1
        1.5 1.5 0.1 0.1
        1.0 1.0 0.2 0.2
    ];
    
    pause(5)

    %Set speed (col. 1, 2) and acc. (col. 3, 4) to first row in above array
    setTreadmill(setpoints(1,1)*1000,setpoints(1,2)*1000,setpoints(1,3)*1000,setpoints(1,4)*1000)
    %Wait for 60 seconds (change this if you like)
    pause(60)
    %Set speed and acc. to second row
    setTreadmill(setpoints(2,1)*1000,setpoints(2,2)*1000,setpoints(2,3)*1000,setpoints(2,4)*1000)
    %etc...
    pause(60)
    setTreadmill(setpoints(3,1)*1000,setpoints(3,2)*1000,setpoints(3,3)*1000,setpoints(3,4)*1000)
    pause(60)
    setTreadmill(setpoints(4,1)*1000,setpoints(4,2)*1000,setpoints(4,3)*1000,setpoints(4,4)*1000)
    pause(60)
    setTreadmill(setpoints(5,1)*1000,setpoints(5,2)*1000,setpoints(5,3)*1000,setpoints(5,4)*1000)
    pause(60)
    setTreadmill(setpoints(6,1)*1000,setpoints(6,2)*1000,setpoints(6,3)*1000,setpoints(6,4)*1000)
    pause(60)
    setTreadmill(setpoints(7,1)*1000,setpoints(7,2)*1000,setpoints(7,3)*1000,setpoints(7,4)*1000)
    pause(60)
    setTreadmill(setpoints(8,1)*1000,setpoints(8,2)*1000,setpoints(8,3)*1000,setpoints(8,4)*1000)
    pause(60)
    setTreadmill(setpoints(9,1)*1000,setpoints(9,2)*1000,setpoints(9,3)*1000,setpoints(9,4)*1000)
    pause(60)
    setTreadmill(setpoints(10,1)*1000,setpoints(10,2)*1000,setpoints(10,3)*1000,setpoints(10,4)*1000)
    pause(60)
    setTreadmill(setpoints(11,1)*1000,setpoints(11,2)*1000,setpoints(11,3)*1000,setpoints(11,4)*1000)
    pause(60)
    setTreadmill(setpoints(12,1)*1000,setpoints(12,2)*1000,setpoints(12,3)*1000,setpoints(12,4)*1000)
    pause(60)
    setTreadmill(setpoints(13,1)*1000,setpoints(13,2)*1000,setpoints(13,3)*1000,setpoints(13,4)*1000)
    pause(60)
    setTreadmill(setpoints(14,1)*1000,setpoints(14,2)*1000,setpoints(14,3)*1000,setpoints(14,4)*1000)
    pause(60)
    setTreadmill(setpoints(15,1)*1000,setpoints(15,2)*1000,setpoints(15,3)*1000,setpoints(15,4)*1000)
    pause(60)
    
    setTreadmill(0, 0,0.1*1000,0.1*1000) %This line stops the treadmill at the end, do not remove
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
    
    [data_f,starttime] = read(d,seconds(17*60 + 10)); %seconds(...) is the length that it records data for. This should match (or be slightly larger than) the total time from the above function
    data_f.Time = data_f.Time + starttime;
    
    allVars = 1:width(data_f);
    newNames = ["FxL" "FyL" "FzL" "MxL" "MyL" "MzL" "ZL" "FxR" "FyR" "FzR" "MxR" "MyR" "MzR" "ZR"]; %w.r.t Y vertical, X a/p, and Z m/l 
    data_f = renamevars(data_f,allVars,newNames);
end