import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load URDF
robot_id = p.loadURDF(
    "/home/home/robot_lab/source/robot_lab/data/Robots/unitree/g1_description/urdf/g1_23dof_rev_1_0.urdf",
    [0, 0, 0], useFixedBase=True)

p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=45,
                             cameraPitch=-30,
                             cameraTargetPosition=[0, 0.5, 0])

torso_index = -1 if p.getBodyInfo(robot_id)[0].decode('utf-8') == "torso_link" else None
if torso_index is None:
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        link_name = p.getJointInfo(robot_id, i)[12].decode('utf-8')
        if link_name == "torso_link":
            torso_index = i
            break
if torso_index is None:
    raise ValueError("Could not find a link named 'torso_link'. Please check your URDF.")

torso_pos, torso_ori = p.getLinkState(robot_id, torso_index)[:2]

def total_com_world(robot_id):
    total_mass = 0.0
    weighted_sum = np.zeros(3)
    num_joints = p.getNumJoints(robot_id)
    
    # Include the base link (-1) and all joints (0 ~ num_joints-1) in the calculation.
    for i in range(-1, num_joints):
        dyn = p.getDynamicsInfo(robot_id, i)
        mass = dyn[0]
        if mass <= 0:
            continue
            
        com_pos = p.getLinkState(robot_id, i, computeLinkVelocity=0, computeForwardKinematics=1)[0]
        weighted_sum += mass * np.array(com_pos)
        total_mass += mass
        
    if total_mass == 0:
        raise ValueError("URDF에 질량 정보가 없습니다. <inertial> 태그를 확인하세요.")
        
    return weighted_sum / total_mass

com_world = total_com_world(robot_id)

# 4. world -> torso_link frame
inv_pos, inv_ori = p.invertTransform(torso_pos, torso_ori)
com_in_torso, _ = p.multiplyTransforms(inv_pos, inv_ori,
                                        com_world, [0, 0, 0, 1])

# ---------------------------------------------------------------
# 5. Visualize the computed COM position with a sphere
# ---------------------------------------------------------------
# Create a red sphere
vis_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                radius=0.03,
                                rgbaColor=[1, 0, 0, 1])

# Place a massless sphere marker at the computed COM position
com_marker_id = p.createMultiBody(baseMass=0,
                                  baseVisualShapeIndex=vis_shape,
                                  basePosition=com_world)

# ---------------------------------------------------------------
# 6. Print results and run simulation
# ---------------------------------------------------------------
print("=== Center of Mass (COM) ===")
print("World frame:", com_world)
print("torso_link frame:", com_in_torso)
print("\n>>> Check the robot and red COM marker in the simulation window. <<<")
print("Simulation will end when you close the window.")

# Keep simulation running for visualization
while p.isConnected():
    p.stepSimulation()
    time.sleep(1./240.) # Smooth animation

p.disconnect()