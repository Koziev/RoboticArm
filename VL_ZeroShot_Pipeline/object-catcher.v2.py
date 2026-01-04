import re
import os
import glob
import math
import pybullet
import time
import json
import numpy as np
import requests
import cv2
import PIL
from PIL import Image
from ikpy.chain import Chain
import transformers
import torch


def dist22(x1, y1, x2, y2) -> float:
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)


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

    def move_to(self, base_roll=None, shoulder_tilt=None, elbow_pitch=None, elbow_roll=None, wrist_pitch=None, gripper_state=None, stop_on_touch=True):
        payload = {}
        if base_roll is not None:
            payload["base_roll"] = base_roll

        if shoulder_tilt is not None:
            payload["shoulder_tilt"] = shoulder_tilt

        if elbow_pitch is not None:
            payload["elbow_pitch"] = elbow_pitch

        if elbow_roll is not None:
            payload["elbow_roll"] = elbow_roll

        if wrist_pitch is not None:
            payload["wrist_pitch"] = wrist_pitch

        if gripper_state is not None:
            payload["gripper_state"] = gripper_state

        payload["stop_on_touch"] = stop_on_touch
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

    def set_gripper(self, gripper_state):
        assert 0 <= gripper_state <= 90
        resp = requests.post(self.url + "/set_gripper", json={"gripper_state": gripper_state})
        return resp.json()


class SceneAnalyser(object):
    def __init__(self):
        # Initialize the model and processor
        # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        #model_id = "/media/inkoziev/corpora/models/Qwen2.5-VL-3B-Instruct"
        model_id = "/media/inkoziev/corpora/models/Qwen2.5-VL-7B-Instruct"

        # Load model and tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Qwen2VLForConditionalGeneration
        self.model = transformers.AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
            device_map="auto",  # Automatically uses GPU if available
            trust_remote_code=True
        )

    def extract_assistant_response(self, full_response):
        """
        Extract only the assistant's response using chat template logic
        """
        # Method 1: Using the chat template's assistant marker
        assistant_marker = "assistant\n"
        if assistant_marker in full_response:
            # Split by assistant marker and take the last part
            parts = full_response.split(assistant_marker)
            if len(parts) > 1:
                # The assistant's response is after the last assistant marker
                assistant_part = parts[-1]
                # Remove any trailing special tokens
                assistant_part = assistant_part.replace("<|im_end|>", "").strip()
                return assistant_part

        # Method 2: Alternative - extract response after last user turn
        # This handles cases where the model doesn't include the assistant marker
        user_marker = "<|im_start|>user"
        if user_marker in full_response:
            parts = full_response.rsplit(user_marker, 1)
            if len(parts) == 2:
                # Response comes after the last user message
                assistant_part = parts[1]
                # Remove the image tag and user content
                assistant_part = assistant_part.split("<|im_end|>")[-1]
                assistant_part = assistant_part.replace("<|im_end|>", "").strip()
                return assistant_part

        # Method 3: Fallback - return cleaned version
        return full_response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

    def analyze_objects(self, image):
        """
        Analyze objects on a white sheet using Qwen-VL
        """

        # Create prompt with specific instructions
        prompt = """You are looking at a white sheet with 1 to 7 small objects (2-6 cm in size each).

    Analyze the image carefully and:
    1. Identify ALL objects visible on the white sheet
    2. For each object, provide a short description including:
       - Color
       - Shape/type (ball, toy figure, etc.)
       - Any distinctive features
    3. Output ONLY a valid JSON array with this exact format:
    [
     {"object_id": 1, "description": "detailed description here"},
     {"object_id": 2, "description": "detailed description here"},
     ...
    ]

    IMPORTANT:
    - Number objects sequentially starting from 1
    - Be precise and concise in descriptions
    - Do not include any text outside the JSON array
    - If no objects are visible, return empty array: []"""

        # Prepare messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Prepare inputs
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=500,  # Adjust based on expected response length
                do_sample=False,  # Use greedy decoding for more consistent JSON
                temperature=0.1,
                top_p=0.9
            )

        # Decode response
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        answer_text = self.extract_assistant_response(generated_text)

        # Extract JSON from response (Qwen-VL often adds explanatory text)
        json_match = re.search(r'\[.*\]', answer_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                # Clean and parse JSON
                json_str = json_str.replace('\n', ' ').replace('\t', ' ')
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw output: {answer_text}")
                return []
        else:
            print("No JSON found in response")
            print(f"Full output: {generated_text}")
            return []

    def filter_object_list(self, object_list, query):
        output_objects = []

        for object in object_list:

            prompt = """Check whether the object with the description OBJECT satisfies the conditions in the user's query.
Output <start>True</start> if the object with the description OBJECT satisfies the conditions in the query.
Output <start>False</start> if the object with the description OBJECT does not satisfy any of the conditions in the query.

OBJECT:
{}

QUERY:
{}
""".format(object["description"], query)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Prepare inputs
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], return_tensors="pt").to(self.model.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500,  # Adjust based on expected response length
                    do_sample=False,  # Use greedy decoding for more consistent JSON
                    temperature=0.1,
                    top_p=0.9
                )

            # Decode response
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            answer_text = self.extract_assistant_response(generated_text)

            m = re.search("<start>(.*)</start>", answer_text)
            if m:
                object_ok = m.group(1)
                if object_ok == "True":
                    output_objects.append(object)

        return output_objects


class BoundingBoxFinder(object):
    def __init__(self):
        # Prepare processor and model
        model_id = "iSEE-Laboratory/llmdet_large"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)
        self.model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def bounding_box(self, frame, object_descr):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        texts = [[object_descr, ]]

        inputs = self.processor(images=pil_image, text=texts, return_tensors="pt").to(self.model.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess outputs
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.4,
            target_sizes=[(pil_image.height, pil_image.width)]
        )

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # Ищем box с максимальным скором
        best_score = -1e6
        best_box = None
        for box, score in zip(boxes, scores):
            if score > best_score:
                best_score = score
                best_box = box

        best_box = best_box.cpu().numpy().astype(int).tolist()
        return best_box

    def draw_bounding_box(self, frame, bounding_box):
        x1, y1, x2, y2 = bounding_box

        # 4. Draw the rectangle
        color = (0, 255, 0)  # Green color (Blue, Green, Red)
        thickness = 3  # 3 pixels thick
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # # --- Optional: Draw a filled rectangle ---
        # filled_color = (255, 0, 0)  # Blue color
        # filled_thickness = -1  # -1 fills the rectangle
        # cv2.rectangle(frame, (x1, y1), (x2, y2), filled_color, filled_thickness)


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


def show_image_file(image_fp: str, window_title: str):
    pil_image = Image.open(image_fp).convert('RGB')
    opencv_image = np.array(pil_image)
    opencv_bgr_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_title, opencv_bgr_image)
    cv2.waitKey(1)


class TrajectoryPlanner(object):
    def __init__(self, data_root_dir):
        self.data_root_dir = data_root_dir

    def plan_trajectory_for_object_grasping(self, object_x, object_y):
        frames = []
        for frame_fp in glob.glob(os.path.join(self.data_root_dir, "camera_frames/*.json")):
            with open(frame_fp) as f:
                frame_data = json.load(f)
                # timestamp_ns = frame_data["timestamp_ns"]
                frames.append(frame_data)

        with open(os.path.join(self.data_root_dir, "compiler_samples/trajectories.json")) as f:
            trajectories = json.load(f)

        trajectories2 = []
        for trajectory in trajectories:
            # позиция проекции захвата
            gripper_x, gripper_y = trajectory["frame"]["gripper"]["center"]

            dist = math.sqrt(dist22(object_x, object_y, gripper_x, gripper_y))
            trajectories2.append((dist, trajectory))

        trajectories2 = sorted(trajectories2, key=lambda z: z[0])

        nearest_sample = trajectories2[0][1]

        # Для отладки - покажем кадр с камеры
        show_image_file(nearest_sample["frame"]["frame_fp"], "Gripper")

        trajectory1 = nearest_sample["action_point"]
        return trajectory1


if __name__ == "__main__":
    print("Program started")

    urdf_path = "../arm8.urdf"

    # ---------------------------------------------------------------
    # Настройка симулятора

    # pybullet.connect(pybullet.GUI)
    #
    # # отобразим координатные оси для удобства ориентирования
    # pybullet.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0])  # X red
    # pybullet.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0])  # Y green
    # pybullet.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1])  # Z blue
    #
    # pybullet.loadURDF(urdf_path, useFixedBase=True)
    #
    # link_name_to_index = {}
    # num_joints = pybullet.getNumJoints(0)
    # for joint_index in range(num_joints):
    #     joint_info = pybullet.getJointInfo(0, joint_index)
    #     # joint_info[12] = link name (bytes)
    #     link_name = joint_info[12].decode("utf-8")
    #     link_name_to_index[link_name] = joint_index
    # gripper_index = link_name_to_index["gripper_link"]

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

    # ---------------------------------------------------------------

    arm = ArmProxy()
    #arm.reset_servos()

    # "base_roll": 40,
    # "shoulder_tilt": 135,
    # "elbow_pitch": 75,
    # "elbow_roll": 90,
    # "wrist_pitch": 50,
    #
    # arm.set_gripper(45)  # раскроем захват

    arm.move_to(base_roll=40,
                shoulder_tilt=90,
                elbow_pitch=110,
                elbow_roll=95,
                wrist_pitch=90,
                gripper_state=45
                )

    # ---------------------------------------------------------------

    cap = cv2.VideoCapture(0)

    for _ in range(50):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)

        # Get keyboard input
        cv2.waitKey(1)
        time.sleep(0.01)

    # -------------------------------------------------------------------------------------

    print("Entering the main processing loop")

    tplanner = TrajectoryPlanner(".")
    #scene_analyzer = SceneAnalyser()

    pass_counter = 0
    while True:
        pass_counter += 1

        for _ in range(100):
            ret, frame = cap.read()
            cv2.imshow("Camera", frame)
            cv2.waitKey(1)

        print(f"pass_counter={pass_counter}")

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        scene_analyzer = SceneAnalyser()

        print("Start analysing the frame...")
        objects = scene_analyzer.analyze_objects(pil_image)
        # Print results
        print(f"Detected {len(objects)} objects:")
        print(json.dumps(objects, indent=2))

        # Делаем фильтрацию списка найденных объектов по введенному пользователем критерию.
        user_query = "Find and move all green objects"
        objects2 = scene_analyzer.filter_object_list(objects, user_query)
        print("List of objects that meet the user criteria:")
        print(json.dumps(objects2, indent=2))

        del scene_analyzer

        if objects2:
            bb_finder = BoundingBoxFinder()

            for object in objects2:
                object_descr = object["description"]  # берем объект в списке.
                print(f"Object to grasp: {object_descr}")
                box = bb_finder.bounding_box(frame, object_descr)
                if box is not None:
                    # Найден подходящий объект.
                    # Получаем координаты центра его bounding box'а

                    bb_finder.draw_bounding_box(frame, box)
                    cv2.imshow("Object detection", frame)

                    for _ in range(20):
                        cv2.waitKey(1)

                    x1, y1, x2, y2 = box
                    object_x = (x1+x2) // 2
                    object_y = (y1+y2) // 2

                    # Подбираем траекторию захвата.
                    grasping_trajectory = tplanner.plan_trajectory_for_object_grasping(object_x, object_y)

                    print("Press a key to continue...")
                    cv2.waitKey(0)

                    # Запускаем захват.

                    arm.move_to(base_roll=grasping_trajectory["base_roll"],
                                shoulder_tilt=grasping_trajectory["shoulder_tilt"],
                                elbow_pitch=grasping_trajectory["elbow_pitch"],
                                elbow_roll=95,
                                wrist_pitch=grasping_trajectory["wrist_pitch"],
                                gripper_state=45
                                )

                    # Закрываем захват
                    arm.set_gripper(80)

                    # Поднимаем руку
                    arm.move_to(  # shoulder_tilt=90,
                        elbow_pitch=110,
                        wrist_pitch=90,
                    )

                    arm.move_to(shoulder_tilt=90
                                )

                    # Перемещаемся в позицию сброса захваченного объекта
                    arm.move_to(base_roll=48)

                    # Опускаем руку
                    arm.move_to(shoulder_tilt=115,
                                elbow_pitch=90,
                                elbow_roll=95,
                                wrist_pitch=80
                                )

                    # Раскрываем захват.
                    arm.set_gripper(20)
                    time.sleep(0.5)

                    break

            del bb_finder

    print("All done :)")

