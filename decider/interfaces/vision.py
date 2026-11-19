# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# 感知接口层。
# 负责把真机视觉话题或仿真器返回的世界状态，
# 统一整理成上层比赛逻辑可直接读取的形式：
# - 自身位置 / 朝向
# - 球的相对位置
# - 球的全局位置
# - 目标历史轨迹

import math
import time
import numpy as np
from collections import deque

# 条件导入 ROS2，保证没有 ROS2 时也能跑仿真模式。
ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, Header
    from sensor_msgs.msg import JointState, Imu
    from geometry_msgs.msg import Quaternion, Twist, Pose2D, Point
    from nav_msgs.msg import Odometry
    from thmos_msgs.msg import VisionDetections, VisionObj, HeadPose
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    ROS_AVAILABLE = True
except ImportError:
    # 未安装 ROS2 时，仅保留仿真模式所需的最小占位类型。
    Node = object
    VisionDetections = None
    Pose2D = None


class Vision(Node):
    """统一感知接口。"""

    def __init__(self, agent):
        self.agent = agent

        self.is_simulation = getattr(agent, "is_simulation", False)

        if self.is_simulation:
            self.logger = agent.get_logger()
            self.logger.info("[Core/Vision] Initializing Vision in Sim Mode")
        else:
            self.logger = agent.get_logger().get_child("vision_node")

        self._ball_pos_in_vis = np.array([0, 0])  #
        self._ball_pos_in_vis_D = np.array([0, 0])
        self._ball_pos_in_vis_I = np.array([0, 0])
        self._ball_pos = np.array([0, 0])  # mm
        self._ball_pos_in_map = np.array([0, 0])  # mm
        self._vision_last_frame_time = 0
        self._last_ball_time = 0
        self.self_pos = np.array([0, 0])
        self.self_yaw = 0
        self._self_pos_accuracy = 0
        self._ball_pos_accuracy = 0
        self.ball_distance = 6000
        self._search_ball_phase = 0
        self._ball_history = deque(maxlen=20)
        self._detected_objects = list()
        self.robots_map = []
        self.teammates_map = []
        self.opponents_map = []
        self.head = 0.0
        self.neck = 0.0

        if self.is_simulation:
            # 仿真模式下不需要 ROS2 订阅与发布。
            self._location_sub = None
            self._vision_sub = None
            self._relocal_pub = None
            return

        # 订阅 QoS：位姿和视觉都使用尽力而为模式，降低延迟。
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )

        relc_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # 可靠传输
            history=HistoryPolicy.KEEP_LAST,  # 只保留最新消息
            depth=1,  # 队列深度为1
        )

        # 真机模式下订阅定位与视觉消息。
        self._location_sub = self.agent.create_subscription(
            Pose2D, "/THMOS/location", self._position_callback, qos_profile
        )

        self._vision_sub = self.agent.create_subscription(
            VisionDetections, "/THMOS/vision/obj_pos", self._vision_callback, qos_profile
        )

        self._relocal_pub = self.agent.create_publisher(
            Pose2D, "/THMOS/relocalization", relc_qos_profile
        )

    def update_from_sim_state(self, state: dict):
        """
        直接用仿真器返回的状态字典更新感知缓存。

        期望输入格式：
        {
            "robots": [
                {"name": "robot" or "rp0", "x": float, "y": float, "theta": float, "team": "red"},
                ...
            ],
            "ball": {"x": float, "y": float, "z": float}
        }

        当前统一采用的坐标约定：
        1. 机器人局部坐标：X 向前，Y 向左。
        2. 地图坐标：原点在场地中心，X 指向对方球门方向，Y 指向左侧边线。
        3. 仿真返回的主坐标系与当前项目定义基本一致，因此红队通常无需额外变换。
        """
        # 从配置中取本地机器人编号和队伍颜色。
        robot_id = self.agent.get_config().get("id", 0)
        team_color = getattr(self.agent, "color", "red")

        # 1. 先在 robots 数组里找到“我自己”。
        robots = state.get("robots", [])
        robot = None
        self.robots_map = []
        self.teammates_map = []
        self.opponents_map = []

        # 红队命名为 robot_rp{id}，蓝队命名为 robot_bp{id}。
        team_prefix = "rp" if team_color == "red" else "bp"
        expected_name = f"robot_{team_prefix}{robot_id}"

        self.logger.info(
            f"[Vision] team={team_color}, id={robot_id}, looking for '{expected_name}'"
        )
        self.logger.info(f"[Vision] Available robots: {[r.get('name') for r in robots]}")

        for idx, r in enumerate(robots):
            raw_x = float(r.get("x", 0.0))
            raw_y = float(r.get("y", 0.0))
            raw_theta = float(r.get("theta", 0.0))
            if team_color == "blue":
                raw_x = -raw_x
                raw_y = -raw_y
                raw_theta = raw_theta + math.pi
            item = {
                "name": r.get("name", f"robot_{idx}"),
                "team": r.get("team"),
                "x": raw_x,
                "y": raw_y,
                "theta": raw_theta,
            }
            self.robots_map.append(item)
            if r.get("team") == team_color:
                self.teammates_map.append(item)
            else:
                self.opponents_map.append(item)

        # 先按精确名字找；找不到再按队伍列表回退；最后才用单机器人回退逻辑。
        if robots:
            for r in robots:
                if r.get("name") == expected_name:
                    robot = r
                    break

            if robot is None:
                team_robots = [r for r in robots if r.get("team") == team_color]
                if robot_id < len(team_robots):
                    robot = team_robots[robot_id]
                    self.logger.info(
                        f"[Vision] 回退：使用 team_robots[{robot_id}] = {robot.get('name')}"
                    )

            if robot is None and len(robots) == 1:
                robot = robots[0]
                self.logger.warning(f"[Vision] 回退：当前仅有 1 台机器人，使用 {robot.get('name')}")

        self.logger.info(f"[Vision] Selected robot: {robot.get('name') if robot else 'None'}")

        if robot:
            # 仿真器中的世界坐标。
            sim_x = robot.get("x", 0.0)
            sim_y = robot.get("y", 0.0)
            sim_theta = robot.get("theta", 0.0)  # radians

            # 蓝队做 180 度镜像，这样蓝队策略就能“像红队一样思考”。
            # 也就是说，无论红蓝，策略都可以把“朝对方球门方向”理解成 +X。
            team_color = getattr(self.agent, "color", "red")
            if team_color == "blue":
                self.logger.debug(f"[Vision] Mirroring coordinates for Blue Team")
                sim_x = -sim_x
                sim_y = -sim_y
                sim_theta = sim_theta + math.pi

            # 写入项目统一的地图坐标缓存。
            self.self_pos = np.array([sim_x, sim_y])

            mos_theta_rad = sim_theta
            self.self_yaw = self.agent.angle_normalize(mos_theta_rad) * 180.0 / math.pi

            self.logger.debug(
                f"[Vision] Updated robot pose: pos={self.self_pos}, yaw={self.self_yaw:.2f}deg"
            )
        else:
            self.logger.warning(f"[Vision] No robot found in state for robot_id={robot_id}")

        # 2. 更新球的位置。
        ball = state.get("ball", {})
        if ball:
            bx, by = ball.get("x"), ball.get("y")

            # 蓝队下球的位置也要做同样镜像，保证两队策略统一。
            team_color = getattr(self.agent, "color", "red")
            if team_color == "blue":
                bx = -bx
                by = -by

            mos_bx = bx
            mos_by = by
            self._ball_pos_in_map = np.array([mos_bx, mos_by])

            # 再把球从地图坐标转成机器人局部坐标，供追球 / 带球状态机使用。
            dx_global = mos_bx - self.self_pos[0]
            dy_global = mos_by - self.self_pos[1]

            yaw_rad = self.self_yaw * math.pi / 180.0
            vec_angle = math.atan2(dy_global, dx_global) - yaw_rad
            dist = math.sqrt(dx_global**2 + dy_global**2)

            forward_comp = dist * math.cos(vec_angle)
            left_comp = dist * math.sin(vec_angle)

            rel_x = forward_comp  # Forward
            rel_y = left_comp  # Left

            self.logger.debug(
                f"[BALL_DEBUG] dx={dx_global:.2f}, dy={dy_global:.2f}, yaw={self.self_yaw:.1f}, vec_ang={math.degrees(vec_angle):.1f}, rel=[{rel_x:.2f}, {rel_y:.2f}]"
            )

            self._ball_pos = np.array([rel_x, rel_y])
            self.ball_distance = dist
            self._last_ball_time = time.time()

            self._detected_objects = [
                {
                    "label": "ball",
                    "relative_pos": self._ball_pos,
                    "absolute_pos": self._ball_pos_in_map,
                    "distance": self.ball_distance,
                    "confidence": 1.0,
                    "bounding_box_center": np.array([0, 0]),
                    "timestamp": time.time(),
                }
            ]
            self._ball_history.append(
                {
                    "pos": self._ball_pos,
                    "pos_in_map": self._ball_pos_in_map,
                    "pos_in_vis": np.array([0, 0]),
                    "distance": self.ball_distance,
                    "timestamp": self._last_ball_time,
                }
            )

    def _position_callback(self, msg):
        self.self_pos = np.array([msg.x, msg.y])
        self.self_yaw = self.agent.angle_normalize(msg.theta) * 180 / np.pi

    def _soccer_real_callback(self, msg):
        self._ball_pos_in_map = np.array([msg.x, msg.y])

    def _vision_callback(self, msg: VisionDetections):
        """处理真机视觉检测结果，并计算目标的相对 / 绝对位置。"""
        self._vision_last_frame_time = time.time()

        self._detected_objects = []
        self._ball_pos_accuracy = 0

        for obj in msg.detected_objects:
            label = obj.label

            if obj.xmin >= obj.xmax or obj.ymin >= obj.ymax:
                self.logger.warning(f"[Core] Invalid bounding box received for {label}")
                continue

            curr_coord = (np.array([obj.xmin, obj.ymin]) + np.array([obj.xmax, obj.ymax])) * 0.5

            position_projection = np.array(obj.position_projection)
            position_projection = position_projection[:2]
            if position_projection.shape != (2,):
                self.logger.warning(
                    f"[Core] Invalid position_projection format for {label}, expected 2D coordinates"
                )
                continue

            if np.isnan(position_projection).any():
                self.logger.warning(f"[Core] NaN values found in position_projection for {label}")
                continue

            relative_pos = position_projection / 1000

            distance = np.linalg.norm(position_projection) / 1000

            # 把相对坐标根据当前自身位姿旋转回全局坐标。
            rotation_matrix = np.array(
                [
                    [np.cos(self.self_yaw / 180 * math.pi), -np.sin(self.self_yaw / 180 * math.pi)],
                    [np.sin(self.self_yaw / 180 * math.pi), np.cos(self.self_yaw / 180 * math.pi)],
                ]
            )

            rotated_relative = rotation_matrix @ relative_pos
            absolute_coord = self.self_pos + rotated_relative

            object_info = {
                "label": label,
                "relative_pos": relative_pos,
                "absolute_pos": absolute_coord,
                "distance": distance,
                "confidence": obj.confidence,
                "bounding_box_center": curr_coord / 1000,
                "bound_left_low": np.array(obj.bound_left_low[:2]) if obj.bound_left_low else None,
                "bound_right_low": np.array(obj.bound_right_low[:2])
                if obj.bound_right_low
                else None,
                "timestamp": time.time(),
            }

            self._detected_objects.append(object_info)

        # 如果这帧里检测到了球，则选置信度最高的那个作为当前球。
        ball_objects = [obj for obj in self._detected_objects if obj["label"] == "ball"]
        if ball_objects:
            best_ball = max(ball_objects, key=lambda b: b["confidence"])

            # 更新球相关的变量
            self._ball_pos = best_ball["relative_pos"]
            self._ball_pos_in_map = best_ball["absolute_pos"]
            self._ball_pos_in_vis = best_ball["bounding_box_center"]
            self._ball_pos_accuracy = best_ball["confidence"]
            self.ball_distance = best_ball["distance"]
            self._last_ball_time = best_ball["timestamp"]
            self._ball_history.append(
                {
                    "pos": best_ball["relative_pos"],
                    "pos_in_map": best_ball["absolute_pos"],
                    "pos_in_vis": best_ball["bounding_box_center"],
                    "distance": best_ball["distance"],
                    "timestamp": best_ball["timestamp"],
                }
            )

    def get_objects(self):
        """返回当前检测到的所有目标。"""
        return self._detected_objects

    def get_ball_pos(self):
        return self._ball_pos

    def get_ball_pos_in_vis(self):
        return self._ball_pos_in_vis

    def get_ball_pos_in_map(self):
        return self._ball_pos_in_map

    def get_if_ball(self):
        """判断当前是否仍认为“球可见”。"""

        # 真机模式下若图像坐标始终为 0，说明当前没有有效检测结果。
        if np.all(self._ball_pos_in_vis == 0) and not self.is_simulation:
            pass

        if np.all(self._ball_pos_in_vis == 0) and not self.is_simulation:
            return False

        # 无论真机还是仿真，最后都用“最近一次看到球的时间”判断是否丢球。
        current_time = time.time()
        if current_time - self._last_ball_time > 0.5:
            return False

        return True

    def get_ball_history(self):
        """返回球的历史位置序列。"""
        return list(self._ball_history)

    def relocate(self, x=0, y=0, theta=0):
        """发送重定位请求。"""
        if self.is_simulation or not ROS_AVAILABLE:
            return

        msg = Pose2D()
        msg.x = float(x)
        msg.y = float(y)  # m
        msg.theta = float(theta)  # degree

        self._relocal_pub.publish(msg)
