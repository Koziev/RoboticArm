import random
import math
import datetime
import pybullet
import os
import pathlib
import time
import pickle
import json
import numpy
import numpy as np
import requests
import cv2
import PIL
from PIL import Image
import threading
import queue
from ikpy.chain import Chain



def dist2(x1, y1, z1, x2, y2, z2) -> float:
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1)


def deg2rad(deg):
    return deg * math.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / math.pi


def set_arm_degrees(robot_id, degrees):
    """
    degrees: list of 5 joint angles in degrees
    """
    for joint_index, deg in enumerate(degrees):
        rad = deg2rad(deg)
        pybullet.resetJointState(robot_id, joint_index, rad)


def convert_real2sim(yaw_roll_real, shoulder_tilt_real, elbow_pitch_real, elbow_roll_real, wrist_pitch_real):
    yaw_roll_sim = yaw_roll_real

    shoulder_tilt_sim = shoulder_tilt_real - 90

    #elbow_pitch_sim = 90 - elbow_pitch_real - shoulder_tilt_sim
    elbow_pitch_sim = 180 - elbow_pitch_real - shoulder_tilt_real

    elbow_roll_sim = elbow_roll_real - 95

    wrist_pitch_sim = 90 - wrist_pitch_real

    return [yaw_roll_sim, shoulder_tilt_sim, elbow_pitch_sim, elbow_roll_sim, wrist_pitch_sim, 0]


def convert_sim2real(S_base_roll, S_shoulder_tilt, S_elbow_pitch, S_elbow_roll, S_wrist_pitch):
    R_base_roll = S_base_roll
    R_shoulder_tilt = S_shoulder_tilt + 90
    R_elbow_pitch = 90 - S_elbow_pitch - S_shoulder_tilt
    R_elbow_roll = S_elbow_roll + 95
    R_wrist_pitch = 90 - S_wrist_pitch
    return R_base_roll, R_shoulder_tilt, R_elbow_pitch, R_elbow_roll, R_wrist_pitch


class ArmProxy(object):
    def __init__(self):
        self.url = "http://192.168.0.111:8000"

    def move_to(self, base_roll=None, shoulder_tilt=None, elbow_pitch=None, elbow_roll=None, wrist_pitch=None, gripper_state=None):
        payload = {}
        payload["base_roll"] = base_roll
        payload["shoulder_tilt"] = shoulder_tilt
        payload["elbow_pitch"] = elbow_pitch
        payload["elbow_roll"] = elbow_roll
        payload["wrist_pitch"] = wrist_pitch
        payload["gripper_state"] = gripper_state
        resp = requests.post(self.url + "/move_to", json=payload)
        touch = resp.json()["touch"]
        return touch

    def reset_servos(self):
        resp = requests.post(self.url + "/reset")
        current_angles = resp.json()["current_angles"]
        return current_angles

    def get_servos(self):
        resp = requests.post(self.url + "/get_servos")
        current_angles = resp.json()["servos"]
        return current_angles


def rng_uniform(a: float, b: float) -> float:
    if a < b:
        return rng.uniform(a, b)
    elif a > b:
        return rng.uniform(b, a)
    else:
        return a



class RedBallDetector:
    def __init__(self, min_area=200):
        self.low1  = np.array([0,   50,  80])
        self.high1 = np.array([10,  255, 255])
        self.low2  = np.array([170, 50,  50])
        self.high2 = np.array([179, 255, 255])
        self.min_area = min_area

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.low1, self.high1)
        mask2 = cv2.inRange(hsv, self.low2, self.high2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Clean up
        mask = cv2.medianBlur(mask, 7)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_area:
            return None, mask

        (x, y), radius = cv2.minEnclosingCircle(largest)
        return {
            "center": (int(x), int(y)),
            "radius": int(radius),
            "area": area
        }, mask


class GripperTracker:
    def __init__(self, hsv_lower, hsv_upper, min_area=150):
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = min_area

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_area:
            return None, mask

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return {
            "center": (cx, cy),
            "area": area
        }, mask


class VisionSystem:
    def __init__(self, gripper_tracker):
        #self.ball_detector = ball_detector
        self.gripper_tracker = gripper_tracker

    def process(self, frame):
        #ball, ball_mask = self.ball_detector.detect(frame)
        grip, grip_mask = self.gripper_tracker.detect(frame)

        return {
            #"ball": ball,
            "gripper": grip,
            #"ball_mask": ball_mask,
            "gripper_mask": grip_mask
        }


def draw_detection(frame, result):
    # if result["ball"]:
    #     c = result["ball"]["center"]
    #     r = result["ball"]["radius"]
    #     cv2.circle(frame, c, r, (0, 255, 0), 2)
    #     cv2.putText(frame, "BALL", c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    if result["gripper"]:
        c = result["gripper"]["center"]
        cv2.circle(frame, c, 8, (255, 0, 0), -1)
        cv2.putText(frame, "GRIPPER", c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)


def pow2(x):
    return x*x


def get_gripper_position(robot_id, gripper_index):
    state = pybullet.getLinkState(
        robot_id,
        gripper_index,
        computeForwardKinematics=True
    )
    return state[4]   # (x, y, z) in meters



def get_tcp_position(chain, joint_angles):
    """
    joint_angles: full joint vector (len = number of links)
    returns: np.array([x, y, z]) in meters

    ⚠️ Important: joint_angles must be full length, including fixed joints (inactive joints are ignored internally).
    """
    fk = chain.forward_kinematics(joint_angles)
    return fk[:3, 3]


def target_downwards(chain, joint_angles, dz):
    """
    dz: positive number in meters (e.g. 0.02)
    """
    tcp = get_tcp_position(chain, joint_angles)
    target_position = tcp.copy()
    target_position[2] -= dz
    return target_position


if __name__ == "__main__":
    print("Program started")

    urdf_path = "../arm8.urdf"

    # ---------------------------------------------------------------
    # Настройка симулятора

    pybullet.connect(pybullet.GUI)

    # отобразим координатные оси для удобства ориентирования
    pybullet.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0])  # X red
    pybullet.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0])  # Y green
    pybullet.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1])  # Z blue

    pybullet.loadURDF(urdf_path, useFixedBase=True)

    link_name_to_index = {}
    num_joints = pybullet.getNumJoints(0)
    for joint_index in range(num_joints):
        joint_info = pybullet.getJointInfo(0, joint_index)
        # joint_info[12] = link name (bytes)
        link_name = joint_info[12].decode("utf-8")
        link_name_to_index[link_name] = joint_index
    gripper_index = link_name_to_index["gripper_link"]

    # List the names of the actuated joints in your URDF, IN ORDER.
    # This tells IKPy exactly which joints to include in the calculation.
    joint_names = [
        "joint_base_yaw",  # Base yaw rotation
        "joint_shoulder_pitch",  # Shoulder tilt
        "joint_elbow_pitch",  # Elbow pitch
        "elbow_roll",  # Elbow roll
        "joint_wrist_pitch"  # Wrist pitch
    ]

    # # Create the kinematic chain using the joint names
    arm_chain = Chain.from_urdf_file(urdf_path, base_elements=["base_link"])

    print("IKPy Chain created successfully.")
    print("Number of joints:", len(arm_chain.links))
    for i, link in enumerate(arm_chain.links):
        print(i, link.name, link.joint_type)

    full_links_mask = [
        False,  # base_link
        True,  # joint_base_yaw
        True,  # joint_shoulder_pitch
        True,  # joint_elbow_pitch
        False,  # <<=== True,  # elbow_roll
        True,  # joint_wrist_pitch
        False,  # fixed gripper joint
        False  # tcp_fixed fixed
    ]
    arm_chain.active_links_mask = full_links_mask

    # ----------------------------------------
    lookup_table = []
    delta_degree = 2

    joint_constraints = [
        [45, 135],  # base roll
        [-15, 45],  # shoulder tilt
        [0, 45],  # elbow pitch
        [0, 0],  # elbow roll - вращение кисти с захватом
        [0, 180],  # wrist pitch
    ]

    for yaw_roll_real in range(45, 135, delta_degree):
        for shoulder_tilt_real in range(110, 145, delta_degree):
            for elbow_pitch_real in range(60, 90, delta_degree):
                for elbow_roll_real in [95]:
                    for wrist_pitch_real in range(45, 90, delta_degree):
                        sim_degrees = convert_real2sim(yaw_roll_real=yaw_roll_real,
                                                       shoulder_tilt_real=shoulder_tilt_real,
                                                       elbow_pitch_real=elbow_pitch_real,
                                                       elbow_roll_real=elbow_roll_real,
                                                       wrist_pitch_real=wrist_pitch_real)
                        robot_id = 0
                        set_arm_degrees(robot_id, sim_degrees)
                        pos = get_gripper_position(robot_id, gripper_index)

                        if -0.01 < pos[2] < 0.08:
                            # Берем только такие состояния, когда захват почти касается поверхности стола.

                            # расстояние от центра основания (в плоскости поверхности стола)
                            base_dist = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1])

                            # физические ограничения на удаление точки касания захвата.
                            if 0.08 < base_dist < 0.350:
                                pos_x = pos[0]
                                pos_y = pos[1]

                                if True:  #0.15 <= abs(pos_x) < 0.35:
                                    lookup_table.append({
                                        "real_angles_deg": {"base_roll": yaw_roll_real,
                                                            "shoulder_tilt": shoulder_tilt_real,
                                                            "elbow_pitch": elbow_pitch_real,
                                                            "elbow_roll": elbow_roll_real,
                                                            "wrist_pitch": wrist_pitch_real},
                                        "sim_angles_deg": {"base_roll": sim_degrees[0],
                                                           "shoulder_tilt": sim_degrees[1],
                                                           "elbow_pitch": sim_degrees[2],
                                                           "elbow_roll": sim_degrees[3],
                                                           "wrist_pitch": sim_degrees[4]},
                                        "gripper_position": [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)]
                                    })

    lookup_table = random.sample(lookup_table, k=len(lookup_table))

    # ---------------------------------------

    arm = ArmProxy()

    rng = numpy.random.default_rng()

    samples = []

    samples_dir = "./samples"
    pathlib.Path(samples_dir).mkdir(exist_ok=True)

    print("Start exproring...")
    it = 0
    while lookup_table:
        it += 1
        print(f"Start probe #{it}")

        probe_point = lookup_table[0]
        lookup_table = lookup_table[1:]

        # В начальное положение.
        home_pose = arm.reset_servos()
        time.sleep(1.0)

        # Исходное состояние
        R_base_roll0 = home_pose["base_roll"]
        R_shoulder_tilt0 = home_pose["shoulder_tilt"]
        R_elbow_pitch0 = home_pose["elbow_pitch"]
        R_elbow_roll0 = home_pose["elbow_roll"]
        R_wrist_pitch0 = home_pose["wrist_pitch"]
        gripper_state0 = rng_uniform(170, 180)

        touch = arm.move_to(R_base_roll0,
                            R_shoulder_tilt0,
                            R_elbow_pitch0,
                            R_elbow_roll0,
                            R_wrist_pitch0,
                            gripper_state0)

        # Отобразим финальную точку на симуляторе.
        sim_degrees = [probe_point["sim_angles_deg"]["base_roll"],
                       probe_point["sim_angles_deg"]["shoulder_tilt"],
                       probe_point["sim_angles_deg"]["elbow_pitch"],
                       probe_point["sim_angles_deg"]["elbow_roll"],
                       probe_point["sim_angles_deg"]["wrist_pitch"],
                       ]
        set_arm_degrees(0, sim_degrees)
        S_target_pos = get_gripper_position(robot_id, gripper_index)  # координаты захвата на симуляторе в целевой точке

        # Реальную финальную точку подаем в манипулятор.
        R_base_roll = probe_point["real_angles_deg"]["base_roll"]
        R_shoulder_tilt = probe_point["real_angles_deg"]["shoulder_tilt"]
        R_elbow_pitch = probe_point["real_angles_deg"]["elbow_pitch"]
        R_elbow_roll = probe_point["real_angles_deg"]["elbow_roll"]
        R_wrist_pitch = probe_point["real_angles_deg"]["wrist_pitch"]
        gripper_state = rng_uniform(160, 175)

        touch = arm.move_to(R_base_roll,
                            R_shoulder_tilt,
                            R_elbow_pitch,
                            R_elbow_roll,
                            R_wrist_pitch,
                            gripper_state)
        time.sleep(4.0)  # ждём, когда механика придёт в финальное положение или коснётся поверхности.
        ts = datetime.datetime.now().strftime("%I:%M:%S on %d.%m.%Y")
        time.sleep(1.0)

        # Сохраняем паттерн - сервосигналы и признак касания.
        sample = {"start": {"base_roll": R_base_roll0,
                            "shoulder_tilt": R_shoulder_tilt0,
                            "elbow_pitch": R_elbow_pitch0,
                            "elbow_roll": R_elbow_roll0,
                            "wrist_pitch": R_wrist_pitch0,
                            "gripper_state": gripper_state0},
                  "finish": {"base_roll": R_base_roll,
                            "shoulder_tilt": R_shoulder_tilt,
                            "elbow_pitch": R_elbow_pitch,
                            "elbow_roll": R_elbow_roll,
                            "wrist_pitch": R_wrist_pitch,
                            "gripper_state": gripper_state},
                  "touch": touch,
                  "timestamp_ns": time.time_ns(),
                  "timestamp": ts
                  }

        sample_dir = os.path.join(samples_dir, ts)
        pathlib.Path(sample_dir).mkdir(exist_ok=True)

        if touch:
            print("*** TOUCH DETECTED ***")

            # Состояние сервоприводов в момент касания - это важная информация, сохраним её.
            R_touch = arm.get_servos()
            sample["R_touch"] = R_touch

            # Если зафиксировано касание, то можно попробовать взять финальную точку немного выше.

            r = dict()
            r["base_roll"] = R_touch["base_roll"]
            r["shoulder_tilt"] = R_touch["shoulder_tilt"] - 2
            r["elbow_pitch"] = R_touch["elbow_pitch"] + 2
            r["elbow_roll"] = R_touch["elbow_roll"]
            r["wrist_pitch"] = R_touch["wrist_pitch"] + 2

            ss = convert_real2sim(r["base_roll"], r["shoulder_tilt"], r["elbow_pitch"], r["elbow_roll"], r["wrist_pitch"])

            s = dict()
            s["base_roll"] = ss[0]
            s["shoulder_tilt"] = ss[1]
            s["elbow_pitch"] = ss[2]
            s["elbow_roll"] = ss[3]
            s["wrist_pitch"] = ss[4]

            # запланируем эту точку на следующую сессию.
            new_probe_point = {"sim_angles_deg": s,
                               "real_angles_deg": r}
            lookup_table.insert(0, new_probe_point)

        if not touch:
            # Если касания нет - надо определить, насколько высоко расположена фактическая точка,
            # попробовав нащупать касание.

            q_current = [
                0.0,  # UNUSED base_link
                deg2rad(probe_point["sim_angles_deg"]["base_roll"]),  # joint_base_yaw
                deg2rad(probe_point["sim_angles_deg"]["shoulder_tilt"]),  # joint_shoulder_pitch
                deg2rad(probe_point["sim_angles_deg"]["elbow_pitch"]),  # joint_elbow_pitch
                deg2rad(probe_point["sim_angles_deg"]["elbow_roll"]),  # elbow_roll
                deg2rad(probe_point["sim_angles_deg"]["wrist_pitch"]),  # joint_wrist_pitch
                0.0,  # UNUSED fixed gripper joint
                0.0  # UNUSED tcp_fixed fixed
            ]

            # Get current TCP position using FK
            fk = arm_chain.forward_kinematics(q_current)
            tcp_pos = fk[:3, 3]

            # Define a downward Cartesian motion
            dz = 0.04 # dz: positive number in meters (e.g. 0.02)
            target_position = tcp_pos.copy()
            target_position[2] -= dz

            # Ограничиваем изменение углов для некоторых сочленений:
            links_mask_bak = arm_chain.active_links_mask
            arm_chain.active_links_mask = [
                False,  # base_link
                False,  # base_yaw ЗАПРЕТИЛИ ПОВОРОТ
                True,   # shoulder pitch
                True,   # elbow pitch
                False,  # elbow roll  ЗАПРЕТИЛИ КРУЧЕНИЕ
                True,   # wrist pitch
                False,  # fixed gripper joint
                False   # tcp_fixed fixed
            ]

            # Solve IK for the new position
            q_solution = arm_chain.inverse_kinematics(
                target_position,
                initial_position=q_current
            )

            fk2 = arm_chain.forward_kinematics(q_solution)
            pos2 = fk2[:3, 3]

            #dq = q_solution - q_current

            # Вернем исходные ограничения на сочленения.
            arm_chain.active_links_mask = links_mask_bak

            # переместим в новое состояние, чтобы проверить касание.
            S_base_roll2 = rad2deg(q_solution[1])
            S_shoulder_tilt2 = rad2deg(q_solution[2])
            S_elbow_pitch2 = rad2deg(q_solution[3])
            S_elbow_roll2 = rad2deg(q_solution[4])
            S_wrist_pitch2 = rad2deg(q_solution[5])

            R2 = convert_sim2real(S_base_roll2,
                                  S_shoulder_tilt2,
                                  S_elbow_pitch2,
                                  S_elbow_roll2,
                                  S_wrist_pitch2)
            R_base_roll2, R_shoulder_tilt2, R_elbow_pitch2, R_elbow_roll2, R_wrist_pitch2 = R2

            touch2 = arm.move_to(R_base_roll2,
                                R_shoulder_tilt2,
                                R_elbow_pitch2,
                                R_elbow_roll2,
                                R_wrist_pitch2,
                                gripper_state)

            if touch2:
                # Посмотрим состояние манипулятора на момент касания.
                R_touch2 = arm.get_servos()

                # Оценим фактическое изменение координаты Z.
                # Для этого пересчитаем в углы симуляции, обновим состояние симулятора
                # и получим координаты захвата в этот момент.
                S_touch2 = convert_real2sim(R_touch2["base_roll"],
                                            R_touch2["shoulder_tilt"],
                                            R_touch2["elbow_pitch"],
                                            R_touch2["elbow_roll"],
                                            R_touch2["wrist_pitch"]
                                            )
                set_arm_degrees(0, S_touch2)

                S_touch_pos2 = get_gripper_position(robot_id, gripper_index) # координаты захвата на симуляторе в момент касания

                dz = S_target_pos[2] - S_touch_pos2[2]  # это фактическая высота положения захвата в целевой точке.
                print(f"dz={dz}")
                sample["Z_by_touch2"] = dz

        with open(os.path.join(sample_dir, "data.json"), "w") as f:
            json.dump(sample, f, indent=True)

    print("All done :)")
