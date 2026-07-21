from pathlib import Path
import mujoco
import mujoco.viewer

# Get the folder containing this script
script_dir = Path(__file__).parent

# Build the full path to scene.xml
scene_path = script_dir / "franka_emika_panda" / "scene.xml"

print("Loading:", scene_path)

model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()