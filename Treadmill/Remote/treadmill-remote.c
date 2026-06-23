/*
Copyright (c) 2009, 2014 Bertec Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list
  of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the Bertec Corporation nor the names of its contributors may
  be used to endorse or promote products derived from this software without specific
  prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

// treadmill-remote.c : Defines the exported functions for the DLL application.
//

#ifdef _WIN32

// WINDOWS
#pragma comment(lib, "ws2_32.lib")
#include <winsock2.h>
#include <ws2tcpip.h>

#else

// POSIX
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <netdb.h>

#endif // _WIN32

#include <assert.h>
#include <string.h>
#include "private.h"
#include "treadmill-remote.h"

#ifdef _WIN32
BOOL wsaInitialized = FALSE;
static WSADATA wsaData;
#endif
static SOCKET sock = INVALID_SOCKET;

#define MSB(word) ((unsigned char)(word>>8))
#define LSB(word) ((unsigned char)(word))
#define PUSHB(byte) (packet[i++] = byte)
#define PUSHW(word) (packet[i++] = MSB(word), packet[i++] = LSB(word))

enum { UDP = 0, TCP = 1 };

static int initialize(char * ip, char * port, int proto)
{
   int rc;
   struct addrinfo * result;
   struct addrinfo hints;

#ifdef _WIN32
   if (! wsaInitialized) {
      rc = WSAStartup(MAKEWORD(2,2), &wsaData);
      if (rc) return TREADMILL_WSA_STARTUP;
      wsaInitialized = TRUE;
   }
#endif

   memset(&hints, 0, sizeof(hints));
   hints.ai_family = AF_INET;
   if (proto == UDP) {
      hints.ai_socktype = SOCK_DGRAM;
   } else {
      hints.ai_socktype = SOCK_STREAM;
   }
   rc = getaddrinfo(ip, port, &hints, &result);
   if (rc) return TREADMILL_ADDRESS;

   sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
   if (sock == INVALID_SOCKET) {
      freeaddrinfo(result);
      return TREADMILL_SOCKET;
   }

   rc = connect(sock, result->ai_addr, (int)result->ai_addrlen);
   freeaddrinfo(result);
   if (rc == SOCKET_ERROR) {
      closesocket(sock);
      sock = INVALID_SOCKET;
      return TREADMILL_CONNECT;
   }

   if (proto == TCP) {
      BOOL trueOpt = TRUE;
      rc = setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char*)&trueOpt, sizeof(trueOpt));
      if (rc == SOCKET_ERROR) {
         closesocket(sock);
         sock = INVALID_SOCKET;
         return TREADMILL_SETSOCKOPT;
      }
   }

   return TREADMILL_OK;
}

TREADMILLREMOTEDLL_API int WINAPI TREADMILL_initialize(char * ip, char * port)
{
   return initialize(ip, port, TCP);
}

TREADMILLREMOTEDLL_API int WINAPI TREADMILL_initializeUDP(char * ip, char * port)
{
   return initialize(ip, port, UDP);
}

TREADMILLREMOTEDLL_API int WINAPI TREADMILL_setSpeed(double left, double right, double acceleration)
{
   return TREADMILL_setSpeed4(left, right, 0.0, 0.0, acceleration);
}

TREADMILLREMOTEDLL_API int WINAPI TREADMILL_setSpeed4(
      double leftFront, double rightFront,
      double leftRear, double rightRear,
      double acceleration)
{
   char packet[64];
   char * ptr = packet;
   int rc, n;
   unsigned short int speed[4];
   unsigned short int accel;
   unsigned short int zero = 0;
   unsigned int i = 0;

   if (sock == INVALID_SOCKET) return TREADMILL_NOT_CONNECTED;

   // ** Incoming Packet format 0 **
   // [u8] format specifier (0)
   // [4 x s16] belt 0..3 speed in mm/s
   // [4 x s16] belt 0..3 acceleration in mm/s^2
   // [s16] incline angle in 0.01 deg
   // [4 x s16] 1s complement (negated) belt 0..3 speed in mm/s
   // [4 x s16] 1s complement (negated) belt 0..3 acceleration in mm/s^2
   // [s16] 1s complement (negated) incline angle in 0.01 deg
   // [27 x u8] padding
   // Size: 1 + 2*(16+2) + 27 = 28+36 = 64 bytes

   speed[0] = (unsigned short)(rightFront * 1000.0);
   speed[1] = (unsigned short)(leftFront * 1000.0);
   speed[2] = (unsigned short)(rightRear * 1000.0);
   speed[3] = (unsigned short)(leftRear * 1000.0);
   accel = (unsigned short)(acceleration * 1000.0);

   memset(packet, 0, sizeof(packet));
   PUSHB(0); // format
   // straight data, then its 1s complement
   for (n = 0; n < 2; ++ n) {
      PUSHW(speed[0]);
      PUSHW(speed[1]);
      PUSHW(speed[2]);
      PUSHW(speed[3]);
      PUSHW(accel);
      PUSHW(accel);
      PUSHW(accel);
      PUSHW(accel);
      PUSHW(zero);
      speed[0] ^= 0xFFFF;
      speed[1] ^= 0xFFFF;
      speed[2] ^= 0xFFFF;
      speed[3] ^= 0xFFFF;
      accel ^= 0xFFFF;
      zero ^= 0xFFFF;
   }
   i += 27;
   assert(i == 64);

   while (ptr < packet+sizeof(packet)) {
      rc = send(sock, packet, sizeof(packet), 0);
      if (rc == SOCKET_ERROR) return TREADMILL_SEND;
      ptr += rc;
   }

   return TREADMILL_OK;
}

TREADMILLREMOTEDLL_API void WINAPI TREADMILL_close(void)
{
   if (sock != INVALID_SOCKET) {
      closesocket(sock);
      sock = INVALID_SOCKET;
   }
}

