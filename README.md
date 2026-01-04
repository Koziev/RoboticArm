## Эксперименты с мультимодальными LLM, CV и физическим ИИ

Начальный сетап экспериментов опирается на простой манипулятор с 5 степенями свободы и захватом, управляемый с помощью цифровых сервоприводов:

Ракурс №1: <img src="KinematicBaseline/20251220_085204.jpg" alt="RoboticArm view 1" width="300" height="200">

Ракурс №2: <img src="KinematicBaseline/20251220_085430.jpg" alt="RoboticArm view 2" width="300" height="200">

### Симуляция

Некоторые задачи на кинематику манипулятора удобно решать не "в живую", а с помощью симулятора, выполняющего геометрические и физические расчеты.
В роли такого симулятора можно взять [PyBullet](https://github.com/bulletphysics/bullet3).
Для симуляции потребуется формальное описание манипулятора - размеры частей, наличие подвижных или неподвижных сочленений, их инерционные свойства.
Стандартным для pybullet подходом является создание URDF файла. Я создал такой файл для используемого манипулятора: [ссылка на URDF файл](KinematicBaseline/arm7.urdf).
Этот файл можно прямо загрузить в небольшом коде:

```python
import pybullet as p
import math


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
        p.resetJointState(robot_id, joint_index, rad)


p.connect(p.GUI)

# отобразим координатные оси для удобства ориентирования
p.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0])  # X red
p.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0])  # Y green
p.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1])  # Z blue

p.loadURDF("arm7.urdf", useFixedBase=True)


def convert_real2sim(yaw_roll_real, shoulder_tilt_real, elbow_pitch_real, elbow_roll_real, wrist_pitch_real):
    yaw_roll_sim = yaw_roll_real

    shoulder_tilt_sim = shoulder_tilt_real - 90

    elbow_pitch_sim = 90 - elbow_pitch_real + (-shoulder_tilt_sim)

    elbow_roll_sim = elbow_roll_real - 95

    wrist_pitch_sim = 90 - wrist_pitch_real

    return [yaw_roll_sim, shoulder_tilt_sim, elbow_pitch_sim, elbow_roll_sim, wrist_pitch_sim, 0]

sim_degrees = convert_real2sim(yaw_roll_real=90,
                               shoulder_tilt_real=130,
                               elbow_pitch_real=90,
                               elbow_roll_real=90,
                               wrist_pitch_real=145)
set_arm_degrees(0, sim_degrees)

input("Press any key...")
```

В появившемся окошке будет интерактивная симуляция, позволяющая установить камеру в удобный ракурс:

<img src="KinematicBaseline/arm simulation 2.png" alt="Simulation view 2" width="300" height="200">

Симулятор позволяет повернуть сервоприводы в заданные углы и получить физические координаты захвата в декартовой системе координат. Используя такой расчет и прогнав таблицу значений углов через симулятор,
можно нахардкодить небольшую проверку работы - см. далее.

### Базовая проверка кинематики

![Grasping and and carrying an object](KinematicBaseline/20251220_090612.gif)


### Пайплайн управления манипулятором на базе Qwen-VL zero-shot prompting

Все компоненты пайплайна собраны в подкаталоге VL_ZeroShot_Pipeline:

1) Обновленный файл URDF с описанием кинематики: [arm8.urdf](VL_ZeroShot_Pipeline/arm8.urdf)

2) На RaspberryPi выполняется веб-сервис, который позволяет через простой rest api давать команды на сервоприводы: [rpi_arm_service.py](VL_ZeroShot_Pipeline/rpi_arm_service.py)

3) На первом этапе происходит калибровка манипулятора через исполнение случайных траекторий и фиксацию изображения манипулятора в конечный момент траектории (когда сервоприводы пришли в целевое состояние или произошло касание захватом поверхности стола). Для калибровки выполняется параллельно 2 кода: запись кадров с камеры [camera_gripper_recorder.py](VL_ZeroShot_Pipeline/camera_gripper_recorder.py) и прогон траекторий [explorer_client_no_camera.v4.py](VL_ZeroShot_Pipeline/explorer_client_no_camera.v4.py). Примерно через 1 час, когда будет отработано около 500 траекторий, с помощью кода [samples_assembler.py](VL_ZeroShot_Pipeline/samples_assembler.py) выполняется сведение записанных изображений и состояний манипулятора.

4) После калибровки выполняется код [object-catcher.v2.py](VL_ZeroShot_Pipeline/object-catcher.v2.py), который а) анализирует изображение с камеры, выделяет с помощью [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) объекты на белом поле перед манипулятором, б) определяет bounding box для одного из найденных объектов с помощью [iSEE-Laboratory/llmdet_large](https://huggingface.co/iSEE-Laboratory/llmdet_large), в) планирует и запускает траекторию для захвата объекта.

Видео с примером удачного захвата:

![Successful grasping with VL model powered pipeline](VL_ZeroShot_Pipeline/green_ball_grasping_success.gif)
