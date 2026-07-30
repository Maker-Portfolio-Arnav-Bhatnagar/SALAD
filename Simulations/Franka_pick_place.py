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

