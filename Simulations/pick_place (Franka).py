#Simulation: Pick & Place (Franka FR3)
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
target_rotation = task.transform_target_to_world.rotation()

cube_position = np.array([0.45, 0.00, 0.02])
lift_position = np.array([0.45, 0.00, 0.30])
drop_position = np.array([0.10, 0.00, 0.02]) 

above_cube    = np.array([0.45, 0.00, 0.10])
target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=above_cube,)
task.set_target(target)

state = "move_above"
wait_counter = 0

#Running the simulation
with mujoco.viewer.launch_passive(model, data) as viewer: #Launches sim

    dt = 0.005 #Controls sim speed
    while viewer.is_running():

        velocity = mink.solve_ik(configuration=configuration,tasks=[task],dt=dt,solver="daqp",) 
        velocity = np.clip(velocity, -0.5, 0.5)

        configuration.integrate_inplace(velocity,dt)
        data.qpos[:7] = configuration.q[:7]

        if state in ["move_above", "move_down", "finished"]:
            data.qpos[7] = 0.04
            data.qpos[8] = 0.04
        else:
            data.qpos[7] = 0.00
            data.qpos[8] = 0.00

        mujoco.mj_forward(model,data)

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

        if state == "move_above" and distance < 0.01:
            target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=cube_position,)
            task.set_target(target)
            state = "move_down"

        elif state == "move_down" and distance < 0.01:
            state = "close_gripper"

        if state == "close_gripper":
            wait_counter += 1
            if wait_counter > 100:
                target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=lift_position,)
                task.set_target(target)
                state = "lift"

        elif state == "lift":
            distance = np.linalg.norm(current_position - lift_position)
            if distance < 0.01:
                print("Lift complete!")
                state = "finished"

        elif state == "lift" and distance < 0.01:
            target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=drop_position,)
            task.set_target(target)
            state = "move_to_drop"

        elif state == "move_to_drop" and distance < 0.01:
            state = "finished"

        viewer.sync()