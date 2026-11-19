# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# 动作接口层。
# 负责把上层状态机产生的“速度 / 转头 / 踢球 / 扑救”请求，
# 翻译成真机 ROS2 消息，或者在仿真模式下写入命令缓存。

import os

# 条件导入 ROS2，保证没有 ROS2 时也能跑仿真模式。
ROS_AVAILABLE = False
try:
    import rclpy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Quaternion, Twist, Pose2D, Point
    from thmos_msgs.msg import VisionDetections, VisionObj
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import Float32MultiArray, Header, Int32

    ROS_AVAILABLE = True
except ImportError:
    # 未安装 ROS2 时，仅保留仿真模式能力。
    pass


class Action:
    """动作执行接口。"""

    def __init__(self, agent):
        self.agent = agent
        # 根据 agent 标志切换真机 / 仿真实现。
        self.is_simulation = getattr(agent, "is_simulation", False)

        if self.is_simulation:
            # 仿真模式下不创建 ROS2 publisher，只保留日志和命令缓存。
            self.logger = agent.get_logger()
            self.logger.info("[Core/Action] Initializing Action in Sim Mode (No ROS)")
            self._cmd_vel_pub = None
            self._head_pose_pub = None
            self._kick_pub = None
            self._save_pub = None
        else:
            # 真机模式下把动作映射到对应 ROS2 话题。
            self.logger = agent.get_logger().get_child("action_node")
            self.logger.info("[Core] Initializing the action module")

            self._cmd_vel_pub = self.agent.create_publisher(Twist, "/THMOS/walk/move", 1)
            self._head_pose_pub = self.agent.create_publisher(
                JointState, "/THMOS/head_control/manual_command", 1
            )
            self._kick_pub = self.agent.create_publisher(Int32, "/THMOS/motion/kick", 1)
            self._save_pub = self.agent.create_publisher(Int32, "/THMOS/motion/save", 1)

    def _move_head(self, pitch, yaw):
        if self.is_simulation:
            # 当前仿真模式下头部控制未真正下发给仿真器。
            return

        head_pose_msg = JointState()
        head_pose_msg.header = Header()
        head_pose_msg.header.stamp = self.agent.get_clock().now().to_msg()
        head_pose_msg.position = [0.0, 0.0]
        head_pose_msg.position[0] = float(yaw)  # Set yaw
        head_pose_msg.position[1] = float(pitch)

        self._head_pose_pub.publish(head_pose_msg)

    def cmd_vel(self, vel_x, vel_y, vel_theta):
        # 统一先转成 float，避免上层传入 numpy 标量等类型。
        vel_x = float(vel_x)
        vel_y = float(vel_y)
        vel_theta = float(vel_theta)

        if self.is_simulation:
            # 仿真模式下不直接发消息，而是更新当前待同步命令。
            self.agent.current_cmd = [vel_x, vel_y, vel_theta]
            return

        cmd = Twist()
        cmd.linear.x = vel_x
        cmd.linear.y = vel_y
        cmd.angular.z = vel_theta
        self._cmd_vel_pub.publish(cmd)

    def do_kick(self, foot=0, death=0):
        """
        执行踢球动作。

        参数：
        - foot: 0 表示右脚，1 表示左脚
        - death: 0 表示普通踢球，1 表示更激进的踢球动作
        """
        if self.is_simulation:
            self.logger.info(f"[Core/Action] Sim Kick: foot={foot}, death={death}")
            self.agent.current_kick = {
                "foot": int(foot),
                "death": int(death),
                "timestamp": float(getattr(self.agent, "sim_now", 0.0) or 0.0),
            }
            return

        table = [["r", "l"], ["r_death", "l_death"]]
        try:
            with open("/tmp/THMOS_kick", "w") as fd:
                fd.write(table[death][foot])
                fd.flush()
            self.logger.info("[Core] Pipe kick command published")
        except Exception as e:
            kick_msg = Int32()
            kick_msg.data = foot + death * 10
            self._kick_pub.publish(kick_msg)
            self.logger.warning("[Core] 已回退到 ROS2 踢球话题")

    def save_ball(self, direction):
        if self.is_simulation:
            self.logger.info(f"[Core/Action] Sim Save: dir={direction}")
            return

        save_msg = Int32()
        save_msg.data = direction
        self._save_pub.publish(save_msg)
        self.logger.info("[Core] Save command published")
