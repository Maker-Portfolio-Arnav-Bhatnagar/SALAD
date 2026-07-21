from pathlib import Path
import mujoco
import mujoco.viewer

script_dir = Path(__file__).parent # Get the folder containing this script
scene_path = script_dir / "franka_emika_panda" / "scene.xml" # Build the full path to scene.xml
print("Loading:", scene_path)

model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

print("\nNumber of joints:", model.njnt)
print("Number of actuators:", model.nu)
print("Number of bodies:", model.nbody)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

