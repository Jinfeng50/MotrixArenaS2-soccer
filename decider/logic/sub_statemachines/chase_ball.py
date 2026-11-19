import math
from math import inf
import time
from transitions import Machine
import numpy as np
import logging


class ChaseBallStateMachine:
    def __init__(self, agent):
        """初始化追球状态机。"""
        self.agent = agent
        self.logger = self.agent.get_logger().get_child("chase_ball_fsm")

        logging.getLogger("transitions").setLevel(logging.WARNING)

        self._config = self.agent.get_config()
        self.read_params()

        # 追球的核心思路很直接：
        # 1. 角度偏差大时先转向
        # 2. 角度偏差小了再前进
        # 3. 距离和角度都满足阈值时认为已到位
        self.states = ["rotate", "forward", "arrived"]
        self.transitions = [
            {
                "trigger": "chase_ball",
                "source": ["rotate", "forward", "arrived"],
                "dest": "arrived",
                "conditions": "close_to_ball",
                "after": "stop_moving_and_set_head",
            },
            {
                "trigger": "chase_ball",
                "source": ["rotate", "forward", "arrived"],
                "dest": "rotate",
                "conditions": ["not_close_to_ball", "large_angle"],
                "after": "rotate_to_ball",
            },
            {
                "trigger": "chase_ball",
                "source": ["rotate", "forward", "arrived"],
                "dest": "forward",
                "conditions": ["not_close_to_ball", "small_angle"],
                "after": "move_forward_to_ball",
            },
        ]

        self.machine = Machine(
            model=self,
            states=self.states,
            initial="rotate",
            transitions=self.transitions,
        )
        self.logger.debug(f"[CHASE BALL FSM] Initialized. Starting state: {self.state}")

    def read_params(self):
        """从配置中读取参数"""
        chase_config = self._config.get("chase", {})
        self.close_angle_threshold_rad = chase_config.get("close_angle_threshold_rad", 0.1)
        self.walk_vel_x = chase_config.get("walk_vel_x", 0.3)
        self.walk_vel_theta = chase_config.get("walk_vel_theta", 0.3)
        self.default_chase_distance = chase_config.get("default_chase_distance", 0.5)
        self.obstacle_avoidance = chase_config.get("obstacle_avoidance", True)
        self.face_to_ball_threshold_degree = chase_config.get("face_to_ball_threshold_degree", 20)

    def close_to_ball(self):
        """判断是否已经到达球附近。

        这里不是只看距离，还要看朝向角是否足够正。
        否则虽然贴近了球，但身体可能还是侧着的，不适合直接带球。
        """
        distance_close = self.chase_distance > self.agent.get_ball_distance()
        if self.agent.get_ball_angle() is None:
            return False
        angle_close = abs(self.agent.get_ball_angle()) < self.close_angle_threshold_rad
        result = distance_close and angle_close
        self.logger.debug(
            f"[CHASE BALL FSM] Close to ball? Distance: {distance_close}, Angle: {angle_close}, Result: {result}"
        )
        return result

    def not_close_to_ball(self):
        """判断是否还没有到球附近。"""
        return not self.close_to_ball()

    def large_angle(self):
        """判断球相对身体前方的夹角是否过大。"""
        if self.agent.get_ball_angle() is None:
            return False
        target_angle_rad = self.agent.get_ball_angle()
        self.logger.debug(
            f"[CHASE BALL FSM] Large angle check: {abs(target_angle_rad) > self.close_angle_threshold_rad}"
        )
        return abs(target_angle_rad) > self.close_angle_threshold_rad

    def small_angle(self):
        """判断当前角度是否已经足够适合前进追球。"""
        if self.agent.get_ball_angle() is None:
            return False
        # target_angle_rad = self.agent.get_ball_angle()
        # return abs(target_angle_rad) < self.close_angle_threshold_rad * 0.5
        return not self.large_angle()

    def run(self, distance=None):
        """执行一拍追球状态机。"""
        self.agent.move_head(inf, inf)
        command = self.agent.get_command().get("command")
        self.logger.debug(f"[CHASE BALL FSM] agent.command: {command}, state: {self.state}")
        # 看不到球时立即停车，避免盲走。
        if not self.agent.get_if_ball():
            self.logger.warning("[CHASE BALL FSM] No ball in sight. Stopping.")
            self.stop_moving()
            return

        self.chase_distance = (
            self.agent.get_command()
            .get("data", {})
            .get("chase_distance", self.default_chase_distance)
        )
        if distance is not None:
            self.chase_distance = distance

        self.logger.debug(f"\n[CHASE BALL FSM] Current state: {self.chase_distance}")

        self.logger.info(f"\n[CHASE BALL FSM] Current state: {self.state}")
        self.logger.debug(f"[CHASE BALL FSM] Triggering 'chase_ball' transition")
        self.machine.model.trigger("chase_ball")

    def rotate_to_ball(self):
        """原地转身，让身体正对球。"""
        self.logger.info("[CHASE BALL FSM] Starting to rotate towards the ball...")
        target_angle_rad = self.agent.get_ball_angle()
        self.logger.info(f"[CHASE BALL FSM] target angle:{target_angle_rad}")
        if self.agent.get_if_ball() == False:
            self.logger.warning("[CHASE BALL FSM] Noball,cant rotate")
            return
        self.agent.cmd_vel(0, 0, np.sign(target_angle_rad) * self.walk_vel_theta)
        self.logger.info("[CHASE BALL FSM] Rotation step completed.")

    def move_forward_to_ball(self):
        """向球前进。"""
        self.logger.debug("[CHASE BALL FSM] Starting to move forward towards the ball...")

        if self.obstacle_avoidance:
            # 若障碍规避模块给出建议速度，则在追球基础上做融合。
            self.logger.debug("[CHASE BALL FSM] Using obstacle avoidance...")
            x_vel, y_vel, theta_vel = self.agent.get_obstacle_avoidance_velocity()
            if x_vel is None or x_vel > self.walk_vel_x:
                x_vel = self.walk_vel_x
            if y_vel is None:
                y_vel = 0.0
            if theta_vel is None:
                self.logger.warning(
                    "[CHASE BALL FSM] Obstacle avoidance failed, using default theta velocity."
                )
                theta_vel = 0.0
        else:
            theta_vel = 0.0
            y_vel = 0.0
            x_vel = self.walk_vel_x

        # 如果已经接近目标追球距离，则不再继续硬顶前进。
        if self.chase_distance > self.agent.get_ball_distance():
            x_vel = 0

        # if x_vel != 0:
        #     y_vel -= -0.03

        self.agent.cmd_vel(
            x_vel,
            y_vel,
            self.agent.get_ball_angle() / np.pi * self.walk_vel_theta * 2 + theta_vel,
        )
        self.logger.debug("[CHASE BALL FSM] Forward movement step completed.")

    def stop_moving(self):
        """停止移动。"""
        self.logger.debug("[CHASE BALL FSM] Stopping movement...")
        self.agent.cmd_vel(0, 0, 0)
        self.logger.debug("[CHASE BALL FSM] Movement stopped.")

    def stop_moving_and_set_head(self):
        """停止移动；若角度仍偏大，则补一个小转向。"""
        target_angle_degree = (
            self.agent.get_ball_angle() * 180 / np.pi
            if self.agent.get_ball_angle() is not None
            else 0
        )
        self.logger.debug("[CHASE BALL FSM] Stopping movement and setting head position...")
        if abs(target_angle_degree) > self.face_to_ball_threshold_degree:
            self.agent.cmd_vel(0, 0, np.sign(target_angle_degree) * self.walk_vel_theta)
        else:
            self.agent.cmd_vel(0, 0, 0)
        self.logger.debug("[CHASE BALL FSM] Movement stopped and head position set.")
