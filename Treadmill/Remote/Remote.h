/*
Copyright (c) 2009, 2013, 2014 Bertec Corporation
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

#ifndef REMOTE_H
#define REMOTE_H

#include <QWidget>
#include <QAbstractSocket>
#include "treadmill-remote.h"
#include "ui_Remote.h"

class Remote : public QWidget
{
   Q_OBJECT
   enum SetpointType { NormalSetpoint, FeedbackRegistrationSetpoint };
public:
   Remote(QWidget *parent = 0, Qt::WindowFlags flags = 0);

private slots:
   void on_disconnect_clicked();
   void on_connect_clicked();
   void on_about_clicked();
   void connected();
   void disconnected();
   void error(QAbstractSocket::SocketError);
   void error(int);
   void readyRead();
   void sendSetpoints(SetpointType t = NormalSetpoint);
   void sendSetpointsDirectly(SetpointType t);
   void sendSetpointsLibrary(SetpointType t);

private:
   Ui::RemoteClass ui;
   QAbstractSocket * socket;
   bool hasLibrary;
   t_TREADMILL_initialize TREADMILL_initialize;
   t_TREADMILL_initializeUDP TREADMILL_initializeUDP;
   t_TREADMILL_setSpeed TREADMILL_setSpeed;
   t_TREADMILL_setSpeed4 TREADMILL_setSpeed4;
   t_TREADMILL_close TREADMILL_close;
};

#endif // REMOTE_H
