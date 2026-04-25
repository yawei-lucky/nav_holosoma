import mujoco

xml_path = "/home/kylin/robotics/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.xml"

model = mujoco.MjModel.from_xml_path(xml_path)

print("ncam =", model.ncam)
for i in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    print(i, name)

# source scripts/source_mujoco_setup.sh
# python test_check_mujoco_camera.py