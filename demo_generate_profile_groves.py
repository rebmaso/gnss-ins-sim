import os
import math
import datetime
import numpy as np
import pandas as pd
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim
from gnss_ins_sim.attitude import attitude

# Constants
D2R = math.pi / 180  # Degrees to radians
R2D = 180 / math.pi  # Radians to degrees

motion_def_path = os.path.abspath('.//demo_motion_def_files//')
fs = 100.0  # IMU sample frequency
fs_gps = 10.0  # GPS sample frequency
fs_mag = fs  # Magnetometer sample frequency, not used for now

script_dir = os.path.dirname(os.path.abspath(__file__))
base_output_dir = os.path.join(script_dir, "demo_saved_data")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir = os.path.join(base_output_dir, timestamp)

os.makedirs(output_dir, exist_ok=True)

def save_trajectory_groves_csv(sim, filename):

    # Extract data from sim
    time = np.array(sim.get_data(["time"]))[0] 
    ref_pos = np.array(sim.get_data(["ref_pos"]))[0]
    ref_vel = np.array(sim.get_data(["ref_vel"]))[0] 
    ref_att_quat = np.array(sim.get_data(["ref_att_quat"]))[0]

    # convert quat to euler in xyz order (Groves)
    ref_att = np.array([attitude.quat2euler(q, rot_seq="xyz") for q in ref_att_quat])
    
    # Convert radian values to degrees
    ref_pos[:, :2] *= R2D
    ref_att *= R2D

    df = pd.DataFrame({
        "time (s)": time,
        "latitude (deg)": ref_pos[:, 0],
        "longitude (deg)": ref_pos[:, 1],
        "height (m)": ref_pos[:, 2],
        "north velocity (m/s)": ref_vel[:, 0],
        "east velocity (m/s)": ref_vel[:, 1],
        "down velocity (m/s)": ref_vel[:, 2],
        "roll (deg)": ref_att[:, 0],
        "pitch (deg)": ref_att[:, 1],
        "yaw (deg)": ref_att[:, 2]
    })

    df.to_csv(filename, index=False, header=False)
    print(f"Groves format Profile.csv saved to {filename}")

def test_path_gen():
    # Choose a built-in IMU model, typical for IMU381
    imu_err = 'mid-accuracy'
    imu = imu_model.IMU(accuracy=imu_err, axis=9, gps=True)

    # Start simulation
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      motion_def_path + "//motion_def-Italy.csv",
                      ref_frame=0,  # Use NED frame to match GROVES format
                      imu=imu,
                      mode=None,
                      env=None,
                      algorithm=None)
    sim.run(1)

    save_trajectory_groves_csv(sim, os.path.join(output_dir, "Profile.csv"))
    sim.plot(['ref_pos', 'ref_att_euler'], opt={'ref_pos': '3d'})

if __name__ == '__main__':
    test_path_gen()
