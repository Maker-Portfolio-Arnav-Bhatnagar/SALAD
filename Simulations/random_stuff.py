import mujoco
import mink
from pathlib import Path

scene_path = Path(r"YOUR_PATH_TO_SCENE.XML")
model = mujoco.MjModel.from_xml_path(str(scene_path))

configuration = mink.Configuration(model)

print([m for m in dir(configuration) if not m.startswith("_")])