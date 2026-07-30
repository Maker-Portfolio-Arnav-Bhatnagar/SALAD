#Simulation: Franka Pick & Place
#Using Mink library & MinkIK to pick & place a simple cube with a Franka FR3 in Mujoco

#Importing libraries
import mujoco
import mujoco.viewer
import mink
import numpy as np
from pathlib import Path
from mink.tasks import FrameTask

#Importing scene model
scene_path = Path(__file__).parent / "franka_emika_panda" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

#Retrieving configuration data
configuration = mink.Configuration(model) #Mink configuration for Franka
configuration.update(data.qpos)
print("Current joint positions:")
print(configuration.q)

#Creating new frametask & setting the target to the cube's location
task = FrameTask(frame_name="ee_site",frame_type="site",position_cost=1.0,orientation_cost=0.1,)
task.set_target_from_configuration(configuration)
target_rotation = (mink.SO3.from_z_radians(np.pi / 4).multiply(task.transform_target_to_world.rotation()))

#Coordinates for positions (X Y Z)
cube_position = np.array([0.45, 0.00, 0.02])
above_cube    = np.array([0.45, 0.00, 0.10])
lift_position = np.array([0.45, 0.00, 0.30])
drop_position = np.array([0.45, 0.30, 0.02])
rest_position = np.array([0.45, 0.20, 0.4])

#Setting initial target
target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=above_cube,) #Tells the robot where to go
task.set_target(target)

state = "move_above" #Basic state tracker
wait_counter = 0

#Running the simulation
with mujoco.viewer.launch_passive(model, data) as viewer: #Launches sim

    dt = 0.005 #Controls sim speed
    while viewer.is_running():

        velocity = mink.solve_ik(configuration=configuration,tasks=[task],dt=dt,solver="daqp",) #Computes joint velocities needed for ee_site to move to target
        velocity = np.clip(velocity, -0.5, 0.5) #Limits max speed for smoother motion

        configuration.integrate_inplace(velocity, dt) #Updates joint positions using computed velocities
        desired_q = configuration.q[:7] #Computes desired joint angles
        
        data.ctrl[:7] = desired_q #Commands MuJoCo actuators to use the desired joint angles

        #Gripper positions
        if state in ["move_above", "move_down", "finished"]:
            data.ctrl[7] = 255  # open
        else:
            data.ctrl[7] = -10  # close

        mujoco.mj_step(model, data)

        #Calculating the distance from actual end effector to its theoretical target
        current_pose = configuration.get_transform_frame_to_world("ee_site", "site")
        current_position = current_pose.translation()
        if state == "move_above":
            distance = np.linalg.norm(current_position - above_cube)
        elif state == "move_down":
            distance = np.linalg.norm(current_position - cube_position)
        elif state == "lift":
            distance = np.linalg.norm(current_position - lift_position)
        elif state == "move_to_drop":
            distance = np.linalg.norm(current_position - drop_position)

        #Reassigns target based on the current state & distance from prior target, then updates the state
        if state == "move_above" and distance < 0.01:
            target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=cube_position,)
            task.set_target(target)
            state = "move_down"

        elif state == "move_down" and distance < 0.01:
            state = "close_gripper"
            print(current_pose.rotation().as_matrix())

        if state == "close_gripper":
            wait_counter += 1
            if wait_counter > 100:
                target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=lift_position,)
                task.set_target(target)
                state = "lift"

        elif state == "lift" and distance < 0.01:
            target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=drop_position,)
            task.set_target(target)
            state = "move_to_drop"

        elif state == "move_to_drop" and distance < 0.01:
            state = "finished"
            print('Finished')  

        if state == "finished":
            target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=rest_position,)
            task.set_target(target)

        viewer.sync()