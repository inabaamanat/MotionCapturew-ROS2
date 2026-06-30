# ROS 2 Treadmill Control System

## Overview

This project is a ROS 2 Jazzy implementation of a human-in-the-loop (HILO) treadmill optimization system for adaptive gait training and biomechanics research. The original Python application has been refactored into a modular ROS architecture, where each package is responsible for a single subsystem.

The system communicates with a Bertec instrumented treadmill, processes gait information, performs self-paced treadmill control, and supports Bayesian optimization for adaptive parameter tuning.

---

## Project Architecture

```
treadmill_ws/
│
├── treadmill_interfaces
│   ├── Custom ROS messages
│   ├── TreadmillCommand.msg
│   ├── TreadmillState.msg
│   └── GaitState.msg
│
├── treadmill_driver
│   ├── Bertec TCP/IP communication
│   ├── Hardware interface
│   └── Publishes treadmill state / receives commands
│
├── treadmill_gait
│   ├── Gait processing algorithms
│   ├── Heel strike detection
│   ├── Symmetry calculations
│   └── Position estimation
│
├── treadmill_controller
│   ├── Self-paced treadmill controller
│   ├── Safety constraints
│   └── Speed command generation
│
├── treadmill_hilo
│   ├── Bayesian optimization
│   ├── Human-in-the-loop parameter tuning
│   └── Cost function evaluation
│
├── treadmill_daq
│   ├── NI-DAQ data acquisition
│   └── Force plate streaming (in progress)
│
└── treadmill_bringup
    └── Launch files for the complete ROS system
```

---

## ROS Data Flow

```
NI-DAQ
   │
   ▼
treadmill_daq
   │
   ▼
treadmill_gait
   │
   ▼
treadmill_controller
   │
   ▼
/treadmill/cmd
   │
   ▼
treadmill_driver
   │
   ▼
Bertec Treadmill
```

---

## Current Progress

### Completed

* ROS 2 workspace setup
* Custom ROS interfaces
* Bertec treadmill TCP/IP driver
* Treadmill command/state messaging
* Gait processing algorithms
* Self-paced controller migration
* Human-in-the-loop Bayesian optimizer migration

### In Progress

* NI-DAQ ROS integration
* Hardware testing with Bertec treadmill
* Windows ↔ WSL communication refinement
* Launch system (`treadmill_bringup`)

---

## Technologies

* ROS 2 Jazzy
* Python
* Bertec Instrumented Treadmill
* NI-DAQmx
* NumPy
* Scikit-Optimize
* Ubuntu (WSL2)

---

## Goals

* Modularize a monolithic biomechanics application into reusable ROS nodes.
* INverse dynamics!
