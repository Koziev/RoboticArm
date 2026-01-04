import time
import math
from adafruit_servokit import ServoKit
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
from typing import Dict, List
import RPi.GPIO as GPIO
import numpy


CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ


def clamp(angle):
    return max(0.0, min(180.0, angle))


def lerp(a, b, t):
    return a + (b - a) * t


class Servo(object):
    def __init__(self, pin: int, kit):
        self.kit = kit
        self.pin = pin
        kit.servo[pin].set_pulse_width_range(500, 2500)
        kit.servo[pin].actuation_range = 180  # 0..180 degrees
        self.cur_angle = None

    def write(self, degrees):
        x = 2.0 + 10.0 * degrees / 180.0
        #self.servo.ChangeDutyCycle(x)
        self.kit.servo[self.pin].angle = degrees
        self.cur_angle = degrees

    def read(self):
        return self.cur_angle

    def stop(self):
        #self.servo.stop()
        pass


class ArmAngles(object):
    base_roll: float
    shoulder_tilt: float
    elbow_pitch: float
    elbow_roll: float
    wrist_pitch: float

    def __init__(self, base_roll: float, shoulder_tilt: float, elbow_pitch: float, elbow_roll: float, wrist_pitch: float):
        self.base_roll = base_roll
        self.shoulder_tilt = shoulder_tilt
        self.elbow_pitch = elbow_pitch
        self.elbow_roll = elbow_roll
        self.wrist_pitch = wrist_pitch

    def __repr__(self):
        return f"base_roll: {self.base_roll}  shoulder_tilt: {self.shoulder_tilt}  elbow_pitch: {self.elbow_pitch}  wrist_pitch: {self.wrist_pitch}"


class RoboticArm(object):
    def __init__(self):
        #GPIO.setmode(GPIO.BOARD)  # Sets the pin numbering system to use the physical layout
        GPIO.setmode(GPIO.BCM)  # Use BCM numbering (GPIO numbers)

        # К этому пину подключен сигнал касания захватом поверхности
        self.touch_signal_pin = 17  # Using GPIO17 (Physical pin 11)
        # Configure pin as input with pull-down resistor
        GPIO.setup(self.touch_signal_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        # Alternatives:
        # GPIO.PUD_UP   - Enable internal pull-up resistor
        # GPIO.PUD_DOWN - Enable internal pull-down resistor
        # None/omit     - No pull resistor (external required)

        # 16-channel PCA9685 board (address 0x40 by default)
        self.kit = ServoKit(channels=16, address=0x40)

        # base yaw
        self.servo2 = Servo(2, self.kit)

        # shoulder tilt
        self.servo4 = Servo(4, self.kit)

        # elbow pitch
        self.servo3 = Servo(3, self.kit)

        # elbow roll
        self.servo5 = Servo(5, self.kit)

        # wrist pitch
        self.servo6 = Servo(6, self.kit)

        # gripper
        self.servo7 = Servo(7, self.kit)

        self.servos = {2: self.servo2,
                       3: self.servo3,
                       4: self.servo4,
                       5: self.servo5,
                       6: self.servo6,
                       7: self.servo7}

        self.reset_servos()

    def reset_servos(self):
        self.servo4.write(90) # shoulder pitch
        self.servo3.write(100) # elbow pitch
        self.servo6.write(90) # wrist pitch
        #time.sleep(0.5)
        self.servo2.write(89) # yaw roll
        self.servo5.write(95) # elbow / wrist roll
        self.servo7.write(170) # gripper

    def close(self):
        for servo in self.servos.values():
            servo.stop()

    def read_torch_signal(self) -> bool:
        """Read and print the current state"""
        state = GPIO.input(self.touch_signal_pin)
        # state will be GPIO.HIGH (1) or GPIO.LOW (0)

        if state == GPIO.HIGH:
            # Сделаем еще несколько чтений, чтобы устранить помехи.
            num_high = 0
            for _ in range(10):
                state = GPIO.input(self.touch_signal_pin)
                if state == GPIO.HIGH:
                    num_high += 1
                time.sleep(0.01)

            if num_high >= 6:
                print(f"DEBUG@150 TOUCH Signal DETECTED: num_high={num_high} - Voltage present")
                return True

        return False

    def set_gripper(self, target_gripper_state):
        if self.read_torch_signal():
            return True

        gripper_state = self.servo7.read()
        delta = 1 if target_gripper_state > gripper_state else -1
        while abs(gripper_state - target_gripper_state) > 0.5:
            gripper_state += delta
            self.servo7.write(gripper_state)
            if self.read_torch_signal():
                return True

        return False

    def turn_smoothly(self, servo, target_angle: int, speed: float, stop_on_touch: bool) -> bool:
        if stop_on_touch and self.read_torch_signal():
            # Касание до начала движения
            return True

        cur_angle = servo.read()
        num_steps = int(abs(target_angle - cur_angle))
        if num_steps > 0:
            delta = (target_angle - cur_angle) / num_steps
            for i in range(num_steps):
                cur_angle += delta
                servo.write(cur_angle)
                if stop_on_touch and self.read_torch_signal():
                    return True

                time.sleep(1e-3/speed)

        return False

    def move_to_angles(self, a: ArmAngles, stop_on_touch: bool) -> bool:
        # TODO: передавать еще порядок включения серводвигателей.
        # Сейчас зашит порядок base_yaw --> shoulder_tilt --> wrist_pitch --> elbow_pitch --> elbow_roll

        if a.base_roll is not None:
            touch_signal = self.turn_smoothly(self.servo2, a.base_roll, speed=0.02, stop_on_touch=stop_on_touch)
            if touch_signal and stop_on_touch:
                return True

        if a.shoulder_tilt is not None:
            touch_signal = self.turn_smoothly(self.servo4, a.shoulder_tilt, speed=0.02, stop_on_touch=stop_on_touch)
            if touch_signal and stop_on_touch:
                return True

        if a.wrist_pitch is not None:
            touch_signal = self.turn_smoothly(self.servo6, a.wrist_pitch, speed=0.02, stop_on_touch=stop_on_touch)
            if touch_signal and stop_on_touch:
                return True

        if a.elbow_pitch is not None:
            touch_signal = self.turn_smoothly(self.servo3, a.elbow_pitch, speed=0.02, stop_on_touch=stop_on_touch)
            if touch_signal and stop_on_touch:
                return True

        if a.elbow_roll is not None:
            touch_signal = self.turn_smoothly(self.servo5, a.elbow_roll, speed=0.02, stop_on_touch=stop_on_touch)
            if touch_signal and stop_on_touch:
                return True

        return False


#
# def sign(x: float) -> float:
#     if x < 0:
#         return -1.0
#     elif x > 0:
#         return 1.0
#     else:
#         return 0.0
#
#
# def move_smooth(target_angles: Dict[int, float], duration: float):
#     sids = list(target_angles.keys())
#     frame = {sid: arm.servos[sid].read() for sid in sids}
#
#     delta_degree = 1.0
#     deltas = {sid: sign(target_angles[sid]-frame[sid])*delta_degree for sid in sids}
#
#     while True:
#         changed = False
#         for sid in sids:
#             if abs(target_angles[sid] - frame[sid]) > 0.5*delta_degree:
#                 frame[sid] += deltas[sid]
#                 changed = True
#
#         if changed:
#             arm.write_atomic(frame)
#             time.sleep(0.01)
#         else:
#             break


app = FastAPI(title="Robotic Arm REST API")

arm = RoboticArm()


# -----------------------------
# Request schema
# -----------------------------
class MoveRequest(BaseModel):
    base_roll: float = None
    shoulder_tilt: float = None
    elbow_pitch: float = None
    elbow_roll: float = None
    wrist_pitch: float = None
    gripper_state: float = None
    stop_on_touch: bool = True


class GetStateRequest(BaseModel):
    servo: int



class SetGripperRequest(BaseModel):
    gripper_state: float = None


# class MoveSmoothRequest(BaseModel):
#     angles: Dict[int, float]
#     duration: float = 0.5  # seconds


# class TrajectoryPoint(BaseModel):
#     angles: Dict[int, float]
#     duration: float
#
#
# class TrajectoryRequest(BaseModel):
#     points: List[TrajectoryPoint]



# -----------------------------
# Helper
# -----------------------------
def clamp(angle):
    return max(0.0, min(180.0, angle))


# -----------------------------
# API endpoint
# -----------------------------
@app.post("/move_to")
def move_to(req: MoveRequest):
    try:
        target_angles = ArmAngles(base_roll=req.base_roll,
                                  shoulder_tilt=req.shoulder_tilt,
                                  elbow_pitch=req.elbow_pitch,
                                  elbow_roll=req.elbow_roll,
                                  wrist_pitch=req.wrist_pitch)
        touch = arm.move_to_angles(target_angles, stop_on_touch=req.stop_on_touch)
        if req.gripper_state is not None and not touch:
            touch = arm.set_gripper(req.gripper_state)

        return {
            "status": "ok",
            "touch": touch,
            "current_angles": {
                "base_roll": arm.servo2.read(),
                "shoulder_tilt": arm.servo4.read(),
                "elbow_pitch": arm.servo3.read(),
                "elbow_roll": arm.servo5.read(),
                "wrist_pitch": arm.servo6.read(),
                "gripper_state": arm.servo7.read(),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_gripper")
def set_gripper(req: SetGripperRequest):
    try:
        touch = arm.set_gripper(req.gripper_state)
        return {
            "status": "ok",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_state")
def get_state(req: GetStateRequest):
    servo_id = req.servo

    if servo_id not in arm.servos:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid servo id {servo_id}. Must be one of {list(arm.servos.keys())}"
        )

    servo = arm.servos[servo_id]
    angle = servo.read()

    return {
        "status": "ok",
        "servo": servo_id,
        "angle": angle
    }


@app.post("/get_servos")
def get_servos():
    servos = {
        "base_roll": arm.servo2.read(),
        "shoulder_tilt": arm.servo4.read(),
        "elbow_pitch": arm.servo3.read(),
        "elbow_roll": arm.servo5.read(),
        "wrist_pitch": arm.servo6.read(),
        "gripper": arm.servo7.read()
    }

    return {
        "status": "ok",
        "servos": servos
    }


# @app.post("/move_smooth")
# def move_smooth_api(req: MoveSmoothRequest):
#     move_smooth(req.angles, req.duration)
#     return {"status": "ok"}


@app.post("/reset")
def reset():
    print("Reset...")
    arm.reset_servos()
    return {"status": "ok",
            "message": "Arm reset to home position",
            "current_angles": {
                "base_roll": arm.servo2.read(),
                "shoulder_tilt": arm.servo4.read(),
                "elbow_pitch": arm.servo3.read(),
                "elbow_roll": arm.servo5.read(),
                "wrist_pitch": arm.servo6.read(),
                "gripper_state": arm.servo7.read(),
            }
            }


# @app.post("/execute_trajectory")
# def execute_trajectory(req: TrajectoryRequest):
#     for point in req.points:
#         move_smooth(point.angles, point.duration)
#
#     return {"status": "ok", "points_executed": len(req.points)}


