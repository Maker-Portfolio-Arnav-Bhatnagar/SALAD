from pathlib import Path
import mujoco
import mujoco.viewer
import math

import mink
import os

print(mink.__file__)
package_dir = os.path.dirname(mink.__file__)
print(package_dir)
print(os.listdir(package_dir))

script_dir = Path(__file__).parent # Get the folder containing this script
scene_path = script_dir / "franka_emika_panda" / "scene.xml" # Build the full path to scene.xml
print("\nLoading:", scene_path)

model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

print("\nNumber of joints:", model.njnt)
print("Number of actuators:", model.nu)
print("Number of bodies:", model.nbody)

print("\nActuators:")
for i in range(model.nu):
    print(i, model.actuator(i).name)

print("\nJoints:")
for i in range(model.njnt):
    print(i, model.joint(i).name)

print("\nNumber of sites:", model.nsite)
print("Sites:")
for i in range(model.nsite):
    print(i, model.site(i).name)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        t = data.time

        data.ctrl[0] = 0.5 * math.sin(t)

        mujoco.mj_step(model, data)
        viewer.sync()

        

