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

cube_position = np.array([0.55, 0.00, 0.25])
above_cube = np.array([0.55, 0.00, 0.30])
at_cube    = np.array([0.55, 0.00, 0.10])
target = mink.SE3.from_rotation_and_translation(rotation=target_rotation,translation=at_cube,)
task.set_target(target)

state = "move_above"

#Running the simulation
with mujoco.viewer.launch_passive(model, data) as viewer: #Launches sim

    dt = 0.005 #Controls sim speed
    while viewer.is_running():

        velocity = mink.solve_ik(configuration=configuration,tasks=[task],dt=dt,solver="daqp",) 
        velocity = np.clip(velocity, -0.5, 0.5)

        configuration.integrate_inplace(velocity,dt)
        data.qpos[:] = configuration.q

        mujoco.mj_forward(model,data)

        viewer.sync()