import os
import sys
import numpy as np
import itertools
import math
sys.path.append("/opt/openrobots/lib/python3.8/site-packages")
from pinocchio import visualize
import pinocchio
import example_robot_data
import crocoddyl
from pinocchio.robot_wrapper import RobotWrapper

# current_directory = os.getcwd()
# print("上层路径：", current_directory)

# change path ??
modelPath = '/home/humanoid_gym/resources/robots/hr_URDF_1023/'
URDF_FILENAME = "urdf/hr_URDF_1023.urdf"
meshPath = '/home/humanoid_gym/resources/robots/hr_URDF_1023/meshes'

# Load the full model
rrobot = RobotWrapper.BuildFromURDF(modelPath + URDF_FILENAME, [meshPath], pinocchio.JointModelFreeFlyer())  # Load URDF file
rmodel = rrobot.model

rightFoot = 'feetr'
leftFoot = 'feetl'

display = crocoddyl.MeshcatDisplay(
    rrobot, frameNames=[rightFoot, leftFoot]
)
q0 = pinocchio.utils.zero(rrobot.model.nq)
print(rrobot.model.nq)
display.display([q0])
#print(1)

rdata = rmodel.createData()
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)

rfId = rmodel.getFrameId(rightFoot)
lfId = rmodel.getFrameId(leftFoot)

rfFootPos0 = rdata.oMf[rfId].translation
lfFootPos0 = rdata.oMf[lfId].translation

comRef = pinocchio.centerOfMass(rmodel, rdata, q0)


#print(1)

initialAngle = np.array([0.38, 0., 0, -0.85, 0.42, 0, 0.38, 0., 0., -0.85, 0.42, 0])
q0 = pinocchio.utils.zero(rrobot.model.nq)
q0[6] = 1  # q.w
q0[2] =0.5848  # z
q0[ 7:20] = initialAngle
display.display([q0])
count = 0

for i in range(rrobot.model.nq-7):
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[i+7] = 1
    display.display([q0])
    #print(1)


for i in range(5000):
    phase = i * 0.005
    sin_pos = np.sin(2 * np.pi * phase)
    sin_pos_l = sin_pos.copy()
    sin_pos_r = sin_pos.copy()

    ref_dof_pos = np.zeros((1,12))
    scale_1 = 0.17
    scale_2 = 2 * scale_1
    # left foot stance phase set to default joint pos
    if sin_pos_l > 0 :
        sin_pos_l = sin_pos_l * 0
    ref_dof_pos[:, 0] = -sin_pos_l * scale_1+initialAngle[0]
    ref_dof_pos[:, 3] = sin_pos_l * scale_2+initialAngle[3]
    ref_dof_pos[:, 4] = -sin_pos_l * scale_1+initialAngle[4]
    # right foot stance phase set to default joint pos
    if sin_pos_r < 0:
        sin_pos_r = sin_pos_r * 0
    ref_dof_pos[:, 6] = sin_pos_r * scale_1+initialAngle[6]
    ref_dof_pos[:, 9] = -sin_pos_r * scale_2+initialAngle[9]
    ref_dof_pos[:, 10] = sin_pos_r * scale_1+initialAngle[10]
    # Double support phase
    ref_dof_pos[np.abs(sin_pos) < 0.1] = ref_dof_pos

    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[7:20] = ref_dof_pos
    # if -0.55 <= sin_pos_l <= -0.45:
    #     # print("ref_dof_pos[:, 0] = -sin_pos_l * scale_1+initialAngle[0] = ",ref_dof_pos[:, 0],"\n")
    #     # print("ref_dof_pos[:, 3] = -sin_pos_l * scale_1+initialAngle[3] = ",ref_dof_pos[:, 3],"\n")
    #     # print("ref_dof_pos[:, 4] = -sin_pos_l * scale_1+initialAngle[4] = ",ref_dof_pos[:, 4],"\n")
    #
    #     # frame_id = rmodel.getFrameId("feetl")
    #     # frame_pose = rdata.oMf[frame_id]  # This is a SE3 object (position and orientation)
    #     # position = frame_pose.translation  # 3D position vector
    #     # rotation = frame_pose.rotation  # 3x3 rotation matrix
    #     # print("top pos: position = ", position, "rotation = ", rotation, "\n\n")
    #
    #     pinocchio.forwardKinematics(rmodel, rdata, q0)
    #     pinocchio.updateFramePlacements(rmodel, rdata)
    #     lf_id = rmodel.getFrameId("feetl")
    #     lf_pos = rdata.oMf[lf_id].translation
    #     print(f"[{i:04d}] feetl position top: z={lf_pos[2]:.4f}")
    #     base_id = rmodel.getFrameId("body")
    #     base_pos = rdata.oMf[base_id].translation
    #     print(f"[{i:04d}] base_pos position top: z={base_pos[2]:.4f}")



# Top of movement
    if -1 <= sin_pos_l <= -0.98:
        # print("ref_dof_pos[:, 0] = -sin_pos_l * scale_1+initialAngle[0] = ",ref_dof_pos[:, 0],"\n")
        # print("ref_dof_pos[:, 3] = -sin_pos_l * scale_1+initialAngle[3] = ",ref_dof_pos[:, 3],"\n")
        # print("ref_dof_pos[:, 4] = -sin_pos_l * scale_1+initialAngle[4] = ",ref_dof_pos[:, 4],"\n")

        # frame_id = rmodel.getFrameId("feetl")
        # frame_pose = rdata.oMf[frame_id]  # This is a SE3 object (position and orientation)
        # position = frame_pose.translation  # 3D position vector
        # rotation = frame_pose.rotation  # 3x3 rotation matrix
        # print("top pos: position = ", position, "rotation = ", rotation, "\n\n")

        pinocchio.forwardKinematics(rmodel, rdata, q0)
        pinocchio.updateFramePlacements(rmodel, rdata)
        lf_id = rmodel.getFrameId("feetl")
        lf_pos = rdata.oMf[lf_id].translation
        print(f"[{i:04d}] feetl position top 11: z={lf_pos[2]:.4f}")
        base_id = rmodel.getFrameId("body")
        base_pos = rdata.oMf[base_id].translation
        print(f"[{i:04d}] base_pos position top: z={base_pos[2]:.4f}")

    if sin_pos_l == 0:
        pinocchio.forwardKinematics(rmodel, rdata, q0)
        pinocchio.updateFramePlacements(rmodel, rdata)
        lf_id_bot = rmodel.getFrameId("feetl")
        lf_pos_bot = rdata.oMf[lf_id_bot].translation
        print(f"[{i:04d}] feetl position bottom: z={lf_pos_bot[2]:.4f}")
        base_id_bot = rmodel.getFrameId("body")
        base_pos_bot = rdata.oMf[base_id_bot].translation
        print(f"[{i:04d}] base_pos position bottom: z={base_pos_bot[2]:.4f}")
        # lk_id_bot = rmodel.getFrameId("kneel")
        # lk_pos_bot = rdata.oMf[lk_id_bot].translation
        # print(f"[{i:04d}] kneel position bottom: z=",lk_pos_bot)
        # rk_id_bot = rmodel.getFrameId("kneer")
        # rk_pos_bot = rdata.oMf[rk_id_bot].translation
        # print(f"[{i:04d}] kneer position bottom: z=",rk_pos_bot)
        # count+=1

    display.display([q0])
    # print(1)
    if count == 2:
        while True:
            pass






for i in range(rrobot.model.nq-7):
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[i+7] = 1
    display.display([q0])
    #print(1)
