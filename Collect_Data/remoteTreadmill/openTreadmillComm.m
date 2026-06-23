function [t] = openTreadmillComm()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


%t=tcpip('localhost',4000);
t = tcpclient("127.0.0.1",4000);
set(t,'InputBufferSize',32,'OutputBufferSize',64);
fopen(t);

end

