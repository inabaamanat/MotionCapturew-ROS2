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

#include "Remote.h"
#include <QMessageBox>
#include <QDataStream>
#include <QTcpSocket>
#include <QUdpSocket>
#include <QLibrary>

Remote::Remote(QWidget *parent, Qt::WindowFlags flags) :
   QWidget(parent, flags),
   socket(0)
{
#if defined(Q_OS_MAC)
   QString libFile = "libtreadmill-remote.dylib";
#elif defined(Q_OS_LINUX)
   QString libFile = "libtreadmill-remote.so";
#elif defined(Q_OS_WIN)
   QString libFile = "treadmill-remote.dll";
#endif
   ui.setupUi(this);
   ui.remoteControls->setDisabled(true);
   ui.disconnect->setDisabled(true);
   ui.useLibrary->setText(ui.useLibrary->text().append(libFile));
   setWindowTitle("Treadmill Remote Control Example 1.0.8");

   connect(ui.exit, SIGNAL(clicked()), SLOT(close()));
   connect(ui.leftFrontSpeedSetpoint, SIGNAL(valueChanged(double)), SLOT(sendSetpoints()));
   connect(ui.rightFrontSpeedSetpoint, SIGNAL(valueChanged(double)), SLOT(sendSetpoints()));
   connect(ui.leftRearSpeedSetpoint, SIGNAL(valueChanged(double)), SLOT(sendSetpoints()));
   connect(ui.rightRearSpeedSetpoint, SIGNAL(valueChanged(double)), SLOT(sendSetpoints()));
   connect(ui.accelerationSetpoint, SIGNAL(valueChanged(double)), SLOT(sendSetpoints()));
   connect(ui.inclineAngleSetpoint, SIGNAL(valueChanged(double)), SLOT(sendSetpoints()));

   QLibrary lib("./treadmill-remote");
   TREADMILL_initialize = (t_TREADMILL_initialize)(lib.resolve("TREADMILL_initialize"));
   TREADMILL_initializeUDP = (t_TREADMILL_initializeUDP)(lib.resolve("TREADMILL_initializeUDP"));
   TREADMILL_setSpeed = (t_TREADMILL_setSpeed)(lib.resolve("TREADMILL_setSpeed"));
   TREADMILL_setSpeed4 = (t_TREADMILL_setSpeed4)(lib.resolve("TREADMILL_setSpeed4"));
   TREADMILL_close = (t_TREADMILL_close)(lib.resolve("TREADMILL_close"));
   hasLibrary = TREADMILL_initialize && TREADMILL_initializeUDP && TREADMILL_setSpeed
          && TREADMILL_setSpeed4 && TREADMILL_close;
   ui.useLibrary->setEnabled(hasLibrary);
}

void Remote::on_connect_clicked()
{
   if (! ui.useLibrary->isChecked()) {
      delete socket;
      if (ui.tcp->isChecked()) socket = new QTcpSocket(this); else socket = new QUdpSocket(this);
      connect(socket, SIGNAL(connected()), SLOT(connected()));
      connect(socket, SIGNAL(disconnected()), SLOT(disconnected()));
      connect(socket, SIGNAL(error(QAbstractSocket::SocketError)), SLOT(error(QAbstractSocket::SocketError)));
      connect(socket, SIGNAL(readyRead()), SLOT(readyRead()));
      socket->connectToHost(ui.host->text(), ui.port->value());
   } else {
      int rc;
      if (ui.tcp->isChecked()) rc = TREADMILL_initialize(ui.host->text().toLatin1().data(), ui.port->text().toLatin1().data());
      else rc = TREADMILL_initializeUDP(ui.host->text().toLatin1().data(), ui.port->text().toLatin1().data());
      if (rc != TREADMILL_OK) {
         error(rc);
         return;
      }
      connected();
   }
}

void Remote::on_disconnect_clicked()
{
   disconnected();
}

void Remote::on_about_clicked()
{
   QMessageBox::about(this,
      "Software License for treadmill-remote",
"Copyright (c) 2009, 2013, 2014 Bertec Corporation\n\
All rights reserved.\n\
\n\
Redistribution and use in source and binary forms, with or without modification, \
are permitted provided that the following conditions are met:\n\
\n\
* Redistributions of source code must retain the above copyright notice, this list \
of conditions and the following disclaimer.\n\
\n\
* Redistributions in binary form must reproduce the above copyright notice, this \
list of conditions and the following disclaimer in the documentation and/or other \
materials provided with the distribution.\n\
\n\
* Neither the name of the Bertec Corporation nor the names of its contributors may \
be used to endorse or promote products derived from this software without specific \
prior written permission.\n\
\n\
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ""AS IS""; \
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE \
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOS \
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE \
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR \
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF \
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS \
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN \
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) \
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE \
POSSIBILITY OF SUCH DAMAGE.\n");
}

void Remote::connected()
{
   if (ui.udp->isChecked()) {
      // Register ourselves for feedback by sending a zero-acceleration setpoint to the
      // treadmill control panel. Such setpoint is otherwise ignored.
      sendSetpoints(FeedbackRegistrationSetpoint);
   }
   else if (! ui.useLibrary->isChecked()) {
      socket->setSocketOption(QAbstractSocket::LowDelayOption, 1);
   }
   ui.connect->setDisabled(true);
   ui.disconnect->setEnabled(true);
   ui.remoteControls->setEnabled(true);
   ui.useLibrary->setDisabled(true);
   ui.tcp->setDisabled(true);
   ui.udp->setDisabled(true);
}

void Remote::disconnected()
{
   if (! ui.useLibrary->isChecked()) {
      socket->close();
   } else {
      TREADMILL_close();
   }
   ui.connect->setEnabled(true);
   ui.disconnect->setDisabled(true);
   ui.remoteControls->setDisabled(true);
   ui.useLibrary->setEnabled(hasLibrary);
   ui.tcp->setEnabled(true);
   ui.udp->setEnabled(true);
}

static void showWarning(QWidget * parent, const QString & title, const QString & text)
{
   Q_ASSERT(parent);
   QMessageBox * box = new QMessageBox(QMessageBox::Warning, title, text, QMessageBox::Ok, parent);
   box->setWindowModality(Qt::WindowModal);
   box->setAttribute(Qt::WA_DeleteOnClose);
   box->show();
}

void Remote::error(QAbstractSocket::SocketError)
{
   showWarning(this, "Network Error", socket->errorString());
   disconnected();
}

void Remote::error(int n)
{
   showWarning(this, "Network Error", QString("Error %1").arg(n));
   disconnected();
}

void Remote::readyRead()
{
   // ** Incoming Packet format 0 **
   // [u8] format specifier (0)
   // [4 x s16] belt 0..3 speed in mm/s
   // [s16] incline angle in 0.01 deg
   // [21 x u8] padding
   // Size: 1 + (8+2) + 21 = 22 + 10 = 32 bytes
   const int packetSize = 32;
   int fullPackets = socket->bytesAvailable() / packetSize;
   if (! fullPackets) return;
   if (fullPackets > 1) socket->read((fullPackets-1) * packetSize);
   const QByteArray data = socket->read(packetSize);
   QDataStream ds(data);
   quint8 format;
   qint16 speed[4];
   qint16 angle;
   ds >> format;
   if (format != 0) return;
   ds >> speed[0];
   ds >> speed[1];
   ds >> speed[2];
   ds >> speed[3];
   ds >> angle;

   ui.rightFrontSpeedFeedback->setText(QString("%1 m/s").arg(speed[0]/1000.0, 0, 'f', 3));
   ui.leftFrontSpeedFeedback->setText(QString("%1 m/s").arg(speed[1]/1000.0, 0, 'f', 3));
   ui.rightRearSpeedFeedback->setText(QString("%1 m/s").arg(speed[2]/1000.0, 0, 'f', 3));
   ui.leftRearSpeedFeedback->setText(QString("%1 m/s").arg(speed[3]/1000.0, 0, 'f', 3));
   ui.inclineAngleFeedback->setText(QString("%1 &deg;").arg(angle/100.0, 0, 'f', 2));
}

void Remote::sendSetpoints(SetpointType t)
{
   if (ui.useLibrary->isChecked()) {
      sendSetpointsLibrary(t);
   } else {
      sendSetpointsDirectly(t);
   }
}

void Remote::sendSetpointsDirectly(SetpointType t)
{
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

   if (socket->state() != QAbstractSocket::ConnectedState) return;
   QByteArray data;
   QDataStream ds(&data, QIODevice::WriteOnly);
   QByteArray filler(27, 0);
   qint16 speed[4];
   qint16 accel;
   qint16 angle;
   speed[0] = ui.rightFrontSpeedSetpoint->value() * 1000.0;
   speed[1] = ui.leftFrontSpeedSetpoint->value() * 1000.0;
   speed[2] = ui.rightRearSpeedSetpoint->value() * 1000.0;
   speed[3] = ui.leftRearSpeedSetpoint->value() * 1000.0;
   accel = (t == NormalSetpoint) ? ui.accelerationSetpoint->value() * 1000.0 : 0.0;
   angle = ui.inclineAngleSetpoint->value() * 100.0;

   ds << (quint8) 0; // format
   // straight
   ds << speed[0];
   ds << speed[1];
   ds << speed[2];
   ds << speed[3];
   ds << accel;
   ds << accel;
   ds << accel;
   ds << accel;
   ds << angle;
   // 1s complement
   ds << (qint16)(speed[0] ^ 0xFFFF);
   ds << (qint16)(speed[1] ^ 0xFFFF);
   ds << (qint16)(speed[2] ^ 0xFFFF);
   ds << (qint16)(speed[3] ^ 0xFFFF);
   ds << (qint16)(accel ^ 0xFFFF);
   ds << (qint16)(accel ^ 0xFFFF);
   ds << (qint16)(accel ^ 0xFFFF);
   ds << (qint16)(accel ^ 0xFFFF);
   ds << (qint16)(angle ^ 0xFFFF);
   ds.writeRawData(filler.data(), filler.size());
   char ldata[64];
   memcpy(ldata, data.data(), 64);
   Q_ASSERT(data.size() == 64);
   socket->write(data);
}

void Remote::sendSetpointsLibrary(SetpointType t)
{
   int rc = TREADMILL_setSpeed4(
            ui.leftFrontSpeedSetpoint->value(),
            ui.rightFrontSpeedSetpoint->value(),
            ui.leftRearSpeedSetpoint->value(),
            ui.rightRearSpeedSetpoint->value(),
            (t == NormalSetpoint) ? ui.accelerationSetpoint->value() : 0);
   if (rc != TREADMILL_OK) error(rc);
}
