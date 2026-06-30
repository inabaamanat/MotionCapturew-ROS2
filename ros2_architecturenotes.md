Inaba Amanat 6/25/2026

DAQ: acquires sensor data.
Gait: interprets walking mechanics.
Controller: decides treadmill commands.
Driver: communicates with the treadmill hardware.
HILO: performs optimization.
Phase 1 - System Architecture

Goal

The goal of this migration is not simply to copy an existing Python project into ROS 2. Instead, the objective is to redesign the software into a modular robotics system where each package has a single responsibility and communicates with the rest of the system using ROS topics and custom message interfaces.

The original implementation relied heavily on a centralized DeviceState object, shared queues, and multithreading. While this worked for a standalone Python application, it tightly coupled all components together. Any changes to one subsystem (GUI, treadmill communication, optimization, etc.) could affect the rest of the program.

ROS 2 encourages a distributed architecture where each node is responsible for one task and exchanges information through well-defined messages. This makes the system easier to debug, extend, and maintain.

System Design

The treadmill software has been divided into independent ROS packages.

Package	Responsibility
treadmill_driver	Communicates directly with the Bertec treadmill hardware through TCP/IP.
treadmill_gait	Processes force plate measurements and detects gait events.
treadmill_controller	Computes treadmill speed commands using the self-paced walking algorithm.
treadmill_hilo	Performs human-in-the-loop optimization and updates optimization parameters.
treadmill_interfaces	Defines all ROS message types used by the system.
treadmill_bringup	Launches the complete ROS system.
Why We Created treadmill_interfaces

Instead of passing NumPy arrays or Python objects between nodes, every subsystem communicates using custom ROS messages.

This provides several advantages:

Every node speaks the same "language."
Messages are self-documenting because each field has a descriptive name.
Nodes remain independent and do not call each other's functions directly.
New hardware or algorithms can be integrated without modifying the rest of the system.
Information Flow
Force Plates
      │
      ▼
treadmill_daq
      │
      ▼
ForcePlateData.msg
      │
      ▼
treadmill_gait
      │
      ▼
GaitState.msg
      │
      ▼
treadmill_controller
      │
      ▼
TreadmillCommand.msg
      │
      ▼
treadmill_driver
      │
      ▼
Bertec Treadmill
      │
      ▼
TreadmillState.msg

The HILO optimizer subscribes to gait and controller information while publishing optimization updates independently.

Design Philosophy

Each ROS package should have exactly one responsibility.

For example:

The driver only knows how to communicate with the treadmill.
The controller only decides what speed the treadmill should move.
The gait node only interprets force plate data.
The optimizer only evaluates walking performance and chooses new parameters.

No package directly calls another package's functions. Instead, all communication occurs through ROS topics and custom messages. This loose coupling makes the software significantly easier to test, debug, and extend