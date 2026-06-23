# Copyright (c) 2009, 2013, Bertec Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this list
#  of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or other
#   materials provided with the distribution.
#
# * Neither the name of the Bertec Corporation nor the names of its contributors may
#   be used to endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

LEVEL = ..
TEMPLATE = app
TARGET = treadmill-remote
QT += network gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
win32-msvc*: QMAKE_LFLAGS *= /IGNORE:4099
CONFIG += windows
contains(QT.core.module_config, ltcg): CONFIG += ltcg

VERSION = 1.0.9.0
QMAKE_TARGET_PRODUCT = "Treadmill Remote Control Example"
QMAKE_TARGET_COPYRIGHT = "Copyright (C) 2009, 2013, 2014 Bertec Corporation"

INCLUDEPATH += $$LEVEL/Treadmill-Remote-Dll
DEPENDPATH += $$LEVEL/Treadmill-Remote-Dll

HEADERS += Remote.h

SOURCES += main.cpp \
    Remote.cpp

FORMS += Remote.ui
