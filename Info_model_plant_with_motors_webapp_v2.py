import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time


# Set wide layout for better use of horizontal space.
st.set_page_config(page_title="Motor Simulation", layout="wide")

# ============================
# Reference Signal Generator
# ============================
def get_speed_reference(cycle, motor_id, cycle_interval=50):
    """
    Returns a speed reference (in RPM) that cycles through four levels.
    The levels are based on the motor_id so that different motors have different sets.
    
    The levels are defined as:
       base, base+100, base+200, base+300,
    where base = (abs(hash(motor_id)) % 200) + 1000.
    The level changes every cycle_interval cycles.
    """
    base = (abs(hash(motor_id)) % 200) + 1000
    levels = [base, base + 100, base + 200, base + 300]
    index = (cycle // cycle_interval) % 4
    return levels[index]

# ============================
# Base Motor Class
# ============================
class Motor:
    """Base class for motor state and common attributes."""
    def __init__(self, motor_id):
        self.motor_id = motor_id
        self.voltage = 0          # [V]
        self.current = 0          # [A]
        self.speed = 0            # [RPM]
        self.torque = 0           # [Nm]
        self.efficiency = 92.0    # [%]
        self.temperature = 25.0   # [°C]
        self.operating_hours = 0  # [cycles]

    def monitor_status(self):
        """Returns current motor status as a dictionary."""
        return {
            "Motor ID": self.motor_id,
            "Voltage (V)": self.voltage,
            "Current (A)": round(self.current, 5),
            "Speed (RPM)": round(self.speed, 10),
            "Torque (Nm)": round(self.torque, 10),
            "Efficiency (%)": round(self.efficiency, 2),
            "Temperature (C)": round(self.temperature, 2),
            "Operating Hours": self.operating_hours
        }

# ============================
# DC Motor Class
# ============================
class DCMotor(Motor):
    """Permanent-magnet DC Motor with closed-loop PI speed control."""
    def __init__(self, motor_id):
        super().__init__(motor_id)
        self.controller_gain = 0.5   # Kp
        self.integral_gain = 0.01    # Ki
        self.integral_error = 0.0

    def control(self, target_speed):
        error = target_speed - self.speed
        self.integral_error += error  # (Assume dt = 1 cycle)
        control_signal = self.controller_gain * error + self.integral_gain * self.integral_error
        control_signal = np.clip(control_signal, 0, 1)
        self.voltage = control_signal * 480
        return control_signal

    def update_motor(self, control_signal):
        max_voltage = 480
        applied_voltage = control_signal * max_voltage
        gain = 3.125        # 480 * 3.125 ≈ 1500 RPM
        time_constant = 10
        self.speed = self.speed + (applied_voltage * gain - self.speed) / time_constant
        self.current = control_signal * 10
        self.torque = self.current * 0.5
        self.temperature += control_signal * 0.1 + random.uniform(-0.05, 0.05)
        self.efficiency -= random.uniform(0, 0.001)
        self.operating_hours += 1

# ============================
# AC Motor Class
# ============================
class ACMotor(Motor):
    """Squirrel-cage AC Motor with closed-loop PI speed control via frequency adjustment."""
    def __init__(self, motor_id):
        super().__init__(motor_id)
        self.base_frequency = 50       # Hz
        self.controller_gain = 0.02    # Kp for frequency
        self.integral_gain = 0.005     # Ki for frequency
        self.integral_error = 0.0
        self.frequency = self.base_frequency

    def control(self, target_speed):
        error = target_speed - self.speed
        self.integral_error += error
        new_frequency = self.base_frequency + self.controller_gain * error + self.integral_gain * self.integral_error
        new_frequency = np.clip(new_frequency, 40, 60)
        self.frequency = new_frequency
        control_signal = new_frequency / 60
        return control_signal

    def update_motor(self, control_signal):
        gain_ac = 30        # 50 Hz * 30 ≈ 1500 RPM
        time_constant = 10
        self.speed = self.speed + (self.frequency * gain_ac - self.speed) / time_constant
        self.current = (self.frequency / 60) * 10
        self.torque = self.current * 0.5
        self.temperature += (self.frequency / 60) * 0.1 + random.uniform(-0.05, 0.05)
        self.efficiency -= random.uniform(0, 0.001)
        self.operating_hours += 1

# ============================
# Machine, Production Line, Factory Classes
# ============================
class Machine:
    """A machine on a production line, with one DC and one AC motor."""
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.dc_motor = DCMotor(f"{machine_id}_DC")
        self.ac_motor = ACMotor(f"{machine_id}_AC")
        self.status = "Running"

    def update_machines(self, cycle, motor_logs):
        # Generate separate reference speeds for each motor.
        target_speed_dc = get_speed_reference(cycle, self.dc_motor.motor_id)
        target_speed_ac = get_speed_reference(cycle, self.ac_motor.motor_id)
        
        # Update DC Motor.
        dc_control_signal = self.dc_motor.control(target_speed_dc)
        self.dc_motor.update_motor(dc_control_signal)
        status_dc = self.dc_motor.monitor_status()
        if self.dc_motor.motor_id not in motor_logs:
            motor_logs[self.dc_motor.motor_id] = {"cycle": [], "speed": [], "temperature": [],
                                                    "efficiency": [], "current": [], "torque": [],
                                                    "reference": []}
        motor_logs[self.dc_motor.motor_id]["cycle"].append(cycle)
        motor_logs[self.dc_motor.motor_id]["speed"].append(status_dc["Speed (RPM)"])
        motor_logs[self.dc_motor.motor_id]["temperature"].append(status_dc["Temperature (C)"])
        motor_logs[self.dc_motor.motor_id]["efficiency"].append(status_dc["Efficiency (%)"])
        motor_logs[self.dc_motor.motor_id]["current"].append(status_dc["Current (A)"])
        motor_logs[self.dc_motor.motor_id]["torque"].append(status_dc["Torque (Nm)"])
        motor_logs[self.dc_motor.motor_id]["reference"].append(target_speed_dc)
        
        # Update AC Motor.
        ac_control_signal = self.ac_motor.control(target_speed_ac)
        self.ac_motor.update_motor(ac_control_signal)
        status_ac = self.ac_motor.monitor_status()
        if self.ac_motor.motor_id not in motor_logs:
            motor_logs[self.ac_motor.motor_id] = {"cycle": [], "speed": [], "temperature": [],
                                                    "efficiency": [], "current": [], "torque": [],
                                                    "reference": []}
        motor_logs[self.ac_motor.motor_id]["cycle"].append(cycle)
        motor_logs[self.ac_motor.motor_id]["speed"].append(status_ac["Speed (RPM)"])
        motor_logs[self.ac_motor.motor_id]["temperature"].append(status_ac["Temperature (C)"])
        motor_logs[self.ac_motor.motor_id]["efficiency"].append(status_ac["Efficiency (%)"])
        motor_logs[self.ac_motor.motor_id]["current"].append(status_ac["Current (A)"])
        motor_logs[self.ac_motor.motor_id]["torque"].append(status_ac["Torque (Nm)"])
        motor_logs[self.ac_motor.motor_id]["reference"].append(target_speed_ac)
        
        return f"Machine {self.machine_id} updated."

class ProductionLine:
    """A production line containing multiple machines."""
    def __init__(self, line_id, num_machines):
        self.line_id = line_id
        self.machines = [Machine(f"{line_id}_M{i}") for i in range(num_machines)]
    
    def update_production_line(self, cycle, motor_logs):
        for machine in self.machines:
            machine.update_machines(cycle, motor_logs)

class Factory:
    """A factory containing multiple production lines."""
    def __init__(self, factory_name, num_lines, machines_per_line):
        self.factory_name = factory_name
        self.production_lines = [ProductionLine(f"Line_{i+1}", machines_per_line) for i in range(num_lines)]

# ============================
# Plotting Functions (Matplotlib)
# ============================

def get_speed_fig_by_machine(motor_logs):
    """
    Create a 2x2 grid figure where each subplot corresponds to one machine.
    Each subplot plots the speed of both DC and AC motors for that machine (with reference).
    The machine grouping is determined by parsing motor IDs.
    """
    # Group motor logs by machine.
    # Expected motor_id format: "Line_{line}_M{machine}_{type}"
    groups = {}
    for motor_id, data in motor_logs.items():
        parts = motor_id.split('_')
        if len(parts) >= 4:
            # Use "Line X - Machine Y" as the key.
            key = f"Line {parts[1]} - {parts[2]}"
            if key not in groups:
                groups[key] = {}
            groups[key][motor_id] = data

    # Expecting 4 groups; sort them.
    keys = sorted(groups.keys())
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axs = axs.flatten()
    for i, key in enumerate(keys):
        ax = axs[i]
        for motor_id, data in groups[key].items():
            ax.plot(data["cycle"], data["speed"], label=motor_id)
            if "reference" in data:
                ax.plot(data["cycle"], data["reference"], label=f"{motor_id} ref", linestyle='--', color='gray')
        ax.set_title(key)
        ax.set_ylabel("Speed (RPM)")
        ax.legend(loc='upper right')
    axs[-1].set_xlabel("Cycle")
    fig.tight_layout()
    return fig

def get_metrics_fig(motor_logs):
    """
    Create a 2x2 grid figure for temperature, efficiency, current, and torque.
    Each subplot shows data for all motors.
    """
    metrics = ["temperature", "efficiency", "current", "torque"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        for motor_id, data in motor_logs.items():
            axs[i].plot(data["cycle"], data[metric], label=motor_id)
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel("Cycle")
    fig.tight_layout()
    return fig

# ============================
# Streamlit App
# ============================
st.title("Real-Time Motor Information Model Simulation")

if st.button("Start Simulation"):
    total_cycles = 101  # Adjust as desired
    dt = 0.1            # Real-time dt (seconds per cycle)
    
    # Create a factory with 2 production lines, each with 2 machines.
    factory = Factory("Smart Information Model Factory", num_lines=2, machines_per_line=2)
    
    # Initialize motor logs dictionary.
    motor_logs = {}
    
    # Create Streamlit placeholders for the two figures and the table.
    col1, col2 = st.columns(2)
    speed_fig_placeholder = col1.empty()
    metrics_fig_placeholder = col2.empty()
    table_placeholder = st.empty()    
 
    for cycle in range(1, total_cycles + 1):
        for line in factory.production_lines:
            line.update_production_line(cycle, motor_logs)
        
        # Update the speed-by-machine figure.
        fig1 = get_speed_fig_by_machine(motor_logs)               
        speed_fig_placeholder.pyplot(fig1)
        
        # Update the metrics 2x2 figure.
        fig2 = get_metrics_fig(motor_logs)
        metrics_fig_placeholder.pyplot(fig2)
                
        # Build and update a table with the latest status for each motor.
        table_data = []
        for motor_id, data in motor_logs.items():
            last_idx = -1
            status = {
                "Motor ID": motor_id,
                "Cycle": data["cycle"][last_idx],
                "Speed (RPM)": data["speed"][last_idx],
                "Temp (°C)": data["temperature"][last_idx],
                "Eff (%)": data["efficiency"][last_idx],
                "Current (A)": data["current"][last_idx],
                "Torque (Nm)": data["torque"][last_idx],
                "Reference": data["reference"][last_idx]
            }
            table_data.append(status)
        df = pd.DataFrame(table_data)
        table_placeholder.dataframe(df)
        
        time.sleep(dt)
    
    st.success("Simulation complete!")
