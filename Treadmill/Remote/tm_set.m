function tm_set(obj, speed, accel, incline)
% TM_SET Sends a treadmill control panel remote control packet.
%    TM_SET(obj, speed, accel) Sets the belt speeds and accelerations.
%    Resets incline position 0.
%
%    TM_SET(obj, speed, accel, incline) Sets the belt speeds,
%    accelerations, and incline position.
%
%    obj is an opened instrument communications object obtained from
%    udp() or tcpip().
%
%    The speed is given in m/s. The acceleration is given in m/s^2.
%    The incline angle is given in degrees above horizontal.
%
%    The speed and accel can be vectors or scalars. If both are vectors,
%    the vectors must be of same size. A scalar value is applied to
%    all belts. A vector value's indices are assigned as follows:
%
%    1 - right (front) belt
%    2 - left (front) belt
%    3 - right rear belt
%    4 - left rear belt
%
%    Values for indices not provided are assumed to be zero speed,
%    and 1 m/s^2 acceleration.
%
%    Values for belts not present on the treadmill are ignored.
%
%    The acceleration must be nonzero. Its sign is ignored.
%
%    See also TCPIP, UDP, FOPEN, FCLOSE.

switch nargin
    case 3
        incline = 0;
    case 4
    otherwise
        error('tm_set must be called with either 3 or 4 arguments');
end

if ~isvector(speed)
    error('speed must be a scalar or a vector');
end

if ~isvector(accel)
    error('accel must be a scalar or a vector');
end

if ~isscalar(speed) && ~isscalar(accel) && length(speed) ~= length(accel)
    error('vector speed and accel must have equal lengths');
end

if isscalar(speed)
    speed = speed * ones(1, 4);
end

if isscalar(accel)
    accel = accel * ones(1, 4);
end

if ~isscalar(incline)
    error('incline angle must be a scalar');
end

if incline < 0
    error('incline angle must be positive');
end

if any(accel == 0)
    error('accel must be non-zero');
end

speed(length(speed)+1:4) = 0;
accel(length(accel)+1:4) = 1;
speed = speed(1:4);
accel = accel(1:4);
accel = abs(accel);

accel(accel > 32.767) = 32.767;
speed(speed > 32.767) = 32.767;
speed(speed < -32.768) = -32.768;
incline(incline > 45) = 45;

packet = zeros(64, 1, 'uint16');
speed = int16(round(1000*speed)); % mm/s
accel = uint16(round(1000*accel)); % mm/s^2
incline = uint16(round(100*incline)); % 0.01deg
speed = typecast(speed, 'uint16');
packet(2+[0:3]*2) = bitshift(speed, -8);
packet(3+[0:3]*2) = bitand(speed, 255);
packet(2+8+[0:3]*2) = bitshift(accel, -8);
packet(3+8+[0:3]*2) = bitand(accel, 255);
packet(2+16) = bitshift(incline, -8);
packet(3+16) = bitand(incline, 255);
packet = uint8(packet);
packet(2+18+[0:17]) = bitxor(packet(2+[0:17]), uint8(255));

fwrite(obj, packet);
end

% This is free and unencumbered software released into the public domain.
% 
% Anyone is free to copy, modify, publish, use, compile, sell, or
% distribute this software, either in source code form or as a compiled
% binary, for any purpose, commercial or non-commercial, and by any
% means.
% 
% In jurisdictions that recognize copyright laws, the author or authors
% of this software dedicate any and all copyright interest in the
% software to the public domain. We make this dedication for the benefit
% of the public at large and to the detriment of our heirs and
% successors. We intend this dedication to be an overt act of
% relinquishment in perpetuity of all present and future rights to this
% software under copyright law.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
% IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
% OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
% ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
% OTHER DEALINGS IN THE SOFTWARE.
% 
% For more information, please refer to <http://unlicense.org/>
