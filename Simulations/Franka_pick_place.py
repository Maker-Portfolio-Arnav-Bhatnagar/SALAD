#Simulation: Franka Pick & Place
#Using a simple Damped Least Squares (DLS) Jacobian IK instead of Mink
 
#Importing libraries
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
 
#Importing scene model
scene_path = Path(__file__).parent / "franka_emika_panda" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
 
#Finding the ID of the end-effector site so we can read its position/orientation each step
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
 
#Settings for the IK controller
POSITION_TOLERANCE = 1    #(m) how close counts as "arrived"
ORIENTATION_TOLERANCE = 0.1  #(rad) how close counts as "arrived"
GAIN = 2.0                    #How aggressively we chase the error each step
MAX_LINEAR_VEL = 0.3          #(m/s) cap on end-effector speed
MAX_ANGULAR_VEL = 0.5         #(rad/s) cap on end-effector rotation speed
DAMPING = 0.05                #Damping factor for the DLS solve, avoids instability near singularities
DT = 0.005                    #Simulation timestep (s)
 
#Gripper actuator values
GRIPPER_OPEN = 255
GRIPPER_CLOSED = -10
 
#Commanded joint angles -- we integrate our own copy instead of reading data.qpos
#back each step, so small tracking lag in the position controller doesn't feed
#back into the IK calculation
q_cmd = data.qpos[:7].copy()
 
 
#Function: computes the joint velocities needed to move the end-effector toward a target pose
def compute_joint_velocity(target_pos, target_quat):
    #Reading the end-effector's current position & orientation
    current_pos = data.site_xpos[ee_site_id].copy()
    current_quat = np.zeros(4)
    mujoco.mju_mat2Quat(current_quat, data.site_xmat[ee_site_id])
 
    #Position error: just target minus current
    pos_error = target_pos - current_pos
 
    #Orientation error: difference between quaternions, converted to a rotation vector
    neg_current_quat = np.zeros(4)
    mujoco.mju_negQuat(neg_current_quat, current_quat)
    quat_error = np.zeros(4)
    mujoco.mju_mulQuat(quat_error, target_quat, neg_current_quat)
    rot_error = np.zeros(3)
    mujoco.mju_quat2Vel(rot_error, quat_error, 1.0)
 
    #Turning the raw error into a capped velocity command (simple P-control)
    lin_vel_cmd = np.clip(GAIN * pos_error, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
    ang_vel_cmd = np.clip(GAIN * rot_error, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
    error = np.concatenate([lin_vel_cmd, ang_vel_cmd])
 
    #Computing the Jacobian: how joint velocities map to end-effector velocity
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    J = np.vstack([jacp[:, :7], jacr[:, :7]])  #Only the 7 arm joints
 
    #Damped least squares solve: turns the 6D end-effector velocity into 7 joint velocities
    JJt = J @ J.T + (DAMPING ** 2) * np.eye(6)
    joint_vel = J.T @ np.linalg.solve(JJt, error)
 
    return joint_vel, np.linalg.norm(pos_error), np.linalg.norm(rot_error)
 
 
#Function: moves the arm to a target position/orientation and waits until it arrives
def move_and_wait(viewer, target_pos, target_quat, name, timeout=8.0):
    global q_cmd
    print(f"Moving to {name}")
    target_pos = np.array(target_pos)
    target_quat = np.array(target_quat)
 
    elapsed = 0.0
    while elapsed < timeout and viewer.is_running():
        joint_vel, pos_dist, rot_dist = compute_joint_velocity(target_pos, target_quat)
 
        #Integrating velocity into our commanded joint angles, then sending to actuators
        q_cmd = q_cmd + joint_vel * DT
        data.ctrl[:7] = q_cmd
 
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed += DT
 
        #Stopping once we are close enough to the target
        if pos_dist < POSITION_TOLERANCE and rot_dist < ORIENTATION_TOLERANCE:
            return
 
    print(f"Timeout reached at {name}")
 
 
#Function: opens or closes the gripper and gives it time to move
def set_gripper(viewer, value, hold_steps=100):
    for _ in range(hold_steps):
        data.ctrl[7] = value
        mujoco.mj_step(model, data)
        viewer.sync()
 
 
#---------------- POSITIONS & ORIENTATIONS ----------------
#Quaternions are in MuJoCo's (w, x, y, z) order
#NOTE: [0, 1, 0, 0] below is a placeholder for "gripper pointing straight down" --
#check what your robot's actual starting quaternion looks like (print current_quat
#from compute_joint_velocity once) and adjust these to match your scene
 
ABOVE_CUBE_POS = [0.45, 0.00, 0.10]
ABOVE_CUBE_QUAT = [0.0, 1.0, 0.0, 0.0]
 
CUBE_POS = [0.45, 0.00, 0.02]
CUBE_QUAT = [0.0, 1.0, 0.0, 0.0]
 
LIFT_POS = [0.45, 0.00, 0.30]
LIFT_QUAT = [0.0, 1.0, 0.0, 0.0]
 
DROP_POS = [0.45, 0.30, 0.02]
DROP_QUAT = [0.0, 1.0, 0.0, 0.0]
 
REST_POS = [0.45, 0.20, 0.40]
REST_QUAT = [0.0, 1.0, 0.0, 0.0]
 
 
#Running the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
 
    #Opening the gripper before we start
    set_gripper(viewer, GRIPPER_OPEN, hold_steps=50)
 
    #Moving above the cube, then down to it
    move_and_wait(viewer, ABOVE_CUBE_POS, ABOVE_CUBE_QUAT, "above cube")
    move_and_wait(viewer, CUBE_POS, CUBE_QUAT, "at cube")
 
    #Closing the gripper to grab the cube
    set_gripper(viewer, GRIPPER_CLOSED, hold_steps=100)
 
    #Lifting, moving to the drop location, and releasing
    move_and_wait(viewer, LIFT_POS, LIFT_QUAT, "lift")
    move_and_wait(viewer, DROP_POS, DROP_QUAT, "drop location")
    set_gripper(viewer, GRIPPER_OPEN, hold_steps=100)
 
    #Returning to a resting position
    move_and_wait(viewer, REST_POS, REST_QUAT, "rest")
 
    print("Sequence complete.")
 
    #Keeping the viewer open so you can look around after finishing
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
