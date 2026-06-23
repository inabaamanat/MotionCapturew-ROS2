#import pprint
###  conda install -n my-env spyder-kernels=2.4
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
import time
import csv
import random, string

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

#import math
plt.clf()

time.sleep(5)

#pp = pprint.PrettyPrinter(indent=4)
runsteps=100000  #~42 per sec 100000=40min
endtime=2*60
#x=0
#run=True
#data_out=[]
#plt.ion()
#i=0


# bertec treadmill setup
ch1=[0]*(runsteps+1)  #FxL
ch2=[0]*(runsteps+1)  #FyR
ch3=[0]*(runsteps+1)  #FzL
ch4=[0]*(runsteps+1)  #MxR
ch5=[0]*(runsteps+1)  #ZeroL
ch6=[0]*(runsteps+1)  #ZeroR
ch7=[0]*(runsteps+1)  #MzR
ch8=[0]*(runsteps+1)  #MxL
ch9=[0]*(runsteps+1)  #MyR
ch10=[0]*(runsteps+1) #FyL
ch11=[0]*(runsteps+1) #FzR
ch12=[0]*(runsteps+1) #MyL
ch13=[0]*(runsteps+1) #GND
ch14=[0]*(runsteps+1) #FxR
ch15=[0]*(runsteps+1) #MzL
ch16=[0]*(runsteps+1) #GND
timesteps=[0]*(runsteps+1)

raw_maxch14=0
raw_maxch3=0
lastLstep=0
lastRstep=0
raw_delR=0
raw_delL=0
L_load_check=[]
L_load_time=[]
L_step_check=[]
L_step_time=[]
R_load_check=[]
R_load_time=[]
R_step_check=[]
R_step_time=[]

lastprint=0
elapsed=0
t=1
run=True
lastmotorupdate = 0
offint = 10
turnofftime=0
pulse=0.3
t_off_rL=0
t_off_rS=0
t_off_lL=0
t_off_lS=0

rLon=False
rSon=False
lLon=False
lSon=False

colors=['black','maroon','red','sandybrown','yellow','yellowgreen','forestgreen','turquoise','cyan','deepskyblue','slategray','royalblue','slateblue','blueviolet','violet','pink']
with nidaqmx.Task() as task:
    
    config = nidaqmx.constants.TerminalConfiguration(10083)
    
    task.ai_channels.add_ai_voltage_chan("Dev1/ai31",terminal_config=config) #Fx LEFT
    task.ai_channels.add_ai_voltage_chan("Dev1/ai23",terminal_config=config) #Fy
    task.ai_channels.add_ai_voltage_chan("Dev1/ai30",terminal_config=config) #Fz
    task.ai_channels.add_ai_voltage_chan("Dev1/ai22",terminal_config=config) #Mx
    task.ai_channels.add_ai_voltage_chan("Dev1/ai29",terminal_config=config) #My
    task.ai_channels.add_ai_voltage_chan("Dev1/ai21",terminal_config=config) #Mz
    task.ai_channels.add_ai_voltage_chan("Dev1/ai28",terminal_config=config) #Z
    task.ai_channels.add_ai_voltage_chan("Dev1/ai20",terminal_config=config) #Fx RIGHT
    task.ai_channels.add_ai_voltage_chan("Dev1/ai27",terminal_config=config) #Fy
    task.ai_channels.add_ai_voltage_chan("Dev1/ai19",terminal_config=config) #Fz
    task.ai_channels.add_ai_voltage_chan("Dev1/ai26",terminal_config=config) #Mx
    task.ai_channels.add_ai_voltage_chan("Dev1/ai18",terminal_config=config) #My
    task.ai_channels.add_ai_voltage_chan("Dev1/ai25",terminal_config=config) #Mz
    task.ai_channels.add_ai_voltage_chan("Dev1/ai17",terminal_config=config) #Z
        
    tzero = time.time()
    while run==True:
        
        data = task.read(number_of_samples_per_channel=1)
        tcurr = time.time()
        elapsed = tcurr - tzero
        
        ch1[t]=data[0][0]
        ch2[t]=data[1][0]
        ch3[t]=data[2][0] ###
        ch4[t]=data[3][0]
        ch5[t]=data[4][0]
        ch6[t]=data[5][0]
        ch7[t]=data[6][0]
        ch8[t]=data[7][0]
        ch9[t]=data[8][0]
        ch10[t]=data[9][0]
        ch11[t]=data[10][0]
        ch12[t]=data[11][0]
        ch13[t]=data[12][0]
        ch14[t]=data[13][0] ###
        ch15[t]=0
        ch16[t]=0
        timesteps[t]=tcurr
        
        if elapsed-lastprint >=1:
            lastprint=int(elapsed)
            print("time: "+str(lastprint))
            if lastprint>=endtime:
                trim=t
                run=False
        
        t+=1
        
######## PLOTTING
channels=[ch1[1:trim], ch2[1:trim], ch3[1:trim], ch4[1:trim], ch5[1:trim], ch6[1:trim], ch7[1:trim], ch8[1:trim], ch9[1:trim], ch10[1:trim], ch11[1:trim], ch12[1:trim], ch13[1:trim], ch14[1:trim], ch15[1:trim], ch16[1:trim], timesteps[1:trim]]
biases=[R_load_check, R_load_time, R_step_check, R_step_time, L_load_check, L_load_time, L_step_check, L_step_time]
#colors=['black','maroon','red','sandybrown','yellow','yellowgreen','forestgreen','turquoise','cyan','deepskyblue','slategray','royalblue','slateblue','blueviolet','violet','pink']

# ch1=[0]*(runsteps+1)  #FxL 0 black
# ch2=[0]*(runsteps+1)  #FyR 1 maroon
# ch3=[0]*(runsteps+1)  #FzL 2 red
# ch4=[0]*(runsteps+1)  #MxR 3 
# ch5=[0]*(runsteps+1)  #ZeroL 4
# ch6=[0]*(runsteps+1)  #ZeroR 5 yellowgreen
# ch7=[0]*(runsteps+1)  #MzR 
# ch8=[0]*(runsteps+1)  #MxL 7
# ch9=[0]*(runsteps+1)  #MyR
# ch10=[0]*(runsteps+1) #FyL 9
# ch11=[0]*(runsteps+1) #FzR 10
# ch12=[0]*(runsteps+1) #MyL
# ch13=[0]*(runsteps+1) #GND
# ch14=[0]*(runsteps+1) #FxR
# ch15=[0]*(runsteps+1) #MzL
# ch16=[0]*(runsteps+1) #GND pink


plt.figure(1)
chosenplots=[2,13] #right
#chosenplots=[0,2,4,7,9,11,14] #left
#chosenplots=[2,13] #

for k in chosenplots:
    #plt.figure(k)
    plt.plot(channels[16],channels[k],c=colors[k])

identity = randomword(10)
### write data to file
with open('treadmill_raw_'+identity+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(channels)
    