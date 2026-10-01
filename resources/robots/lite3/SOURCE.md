Lite3 URDF and STL assets originate from
https://github.com/DeepRoboticsLab/deep_robotics_model at commit
75824b516ecc8fa3f8f7ce5d6577e1e86d6b614f.
The complete upstream BSD 3-Clause license is retained in LICENSE.txt.
No robot assets are loaded from the other implementation's directory.

The asset declares hip torque limits of 24 Nm and knee limits of 36 Nm.
The PIE paper reports peak knee torque 30.5 Nm. The environment clamps every
joint to the smaller of its URDF effort limit and the configured 30.5 Nm limit.
The stand pose [0, -0.8, 1.6] rad per leg is an implementation choice compatible
with the URDF axes and joint limits; it is not specified in the paper.
