import math
from math import inf
import time
from transitions import Machine


class KickStateMachine:
    def __init__(self, agent):
        """初始化踢球状态机。"""
        self.agent = agent
        self.logger = self.agent.get_logger().get_child("kick_fsm")
        self._config = self.agent.get_config()
        self.read_params()  # 初始化参数

        # 踢球分三步对位：
        # 1. angle_adjust：先让身体朝向球门方向
        # 2. back_forth_adjust：把球调整到脚前合适前后距离
        # 3. horizontal_adjust：把球横向挪到左右脚可踢的区域
        self.states = ["angle_adjust", "horizontal_adjust", "back_forth_adjust", "finished"]
        self.transitions = [
            {
                "trigger": "adjust_position",
                "source": ["angle_adjust"],
                "dest": "angle_adjust",
                "conditions": "not_good_angle",
                "after": ["adjust_angle"],
            },
            {
                "trigger": "adjust_position",
                "source": ["horizontal_adjust", "back_forth_adjust"],
                "dest": "angle_adjust",
                "conditions": "really_not_good_angle",
                "after": ["stop", "adjust_angle"],
            },
            {
                "trigger": "adjust_position",
                "source": "angle_adjust",
                "dest": "back_forth_adjust",
                "conditions": "good_angle",
                "after": "stop",
            },
            {
                "trigger": "adjust_position",
                "source": "back_forth_adjust",
                "dest": "back_forth_adjust",
                "conditions": "not_good_back_forth",
                "after": "adjust_back_forth",
            },
            {
                "trigger": "adjust_position",
                "source": "back_forth_adjust",
                "dest": "horizontal_adjust",
                "conditions": "good_back_forth",
                "after": "stop",
            },
            {
                "trigger": "adjust_position",
                "source": ["horizontal_adjust"],
                "dest": "horizontal_adjust",
                "conditions": "not_good_position_horizontally",
                "after": "adjust_horizontally",
            },
            {
                "trigger": "adjust_position",
                "source": "horizontal_adjust",
                "dest": "finished",
                "conditions": "good_position_horizontally",
            },
            {"trigger": "finish", "source": "finished", "dest": "angle_adjust", "after": "stop"},
        ]

        self.machine = Machine(
            model=self,
            states=self.states,
            initial="angle_adjust",
            transitions=self.transitions,
        )
        self.logger.debug(f"[KICK FSM] Initialized. Starting state: {self.state}")

    def run(self, aim_theta=None):
        """执行一拍踢球状态机。"""
        self.logger.debug("[KICK FSM] Starting kick sequence...")
        self.logger.info(f"[KICK FSM] if_ball: {self.agent.get_if_ball()} state: {self.state}")
        if not self.agent.get_if_ball():
            return

        self.agent.move_head(inf, inf)

        # 以“朝球门方向”为默认踢球目标角度。
        field_length = (
            self.agent.get_config().get("field_size", {}).get(self.agent.league, [14.0, 9.0])[0]
        )
        if self.agent.get_self_pos()[0] > field_length / 2 * self.blind_kick_thres_percent:
            target_angle_rad = 0.0  # 已非常接近球门时，直接按正前方处理
        else:
            target_angle_rad = math.atan2(
                -self.agent.get_self_pos()[1], field_length / 2 - self.agent.get_self_pos()[0]
            )

        if aim_theta is not None:
            target_angle_rad = aim_theta / 180 * math.pi

        self.ang_tar = math.degrees(self.agent.angle_normalize(target_angle_rad))
        self.ang_delta = (
            self.agent.angle_normalize((self.ang_tar - self.agent.get_self_yaw()) / 180 * math.pi)
            * 180
            / math.pi
        )
        if (self.agent.get_ball_distance() > self.lost_ball_distance_threshold_m) and (
            self.state != "finished"
        ):
            self.logger.debug("[KICK FSM] Ball is too far. Stopping robot.")
            self.agent.stop()
            return

        if self.state != "finished" and self.agent.get_if_ball():
            self.logger.debug(f"\n[KICK FSM] Current state: {self.state}")
            self.logger.debug(f"[KICK FSM] Triggering 'adjust_position' transition")
            self.adjust_position()

        if self.state == "finished":
            self.logger.debug("\n[KICK FSM] Positioning complete! Executing kick...")
            self.agent.cmd_vel(0, 0, 0)
            # camera_bias 用来补偿视觉偏置，避免“看着正中，实际脚位偏了”。
            ball_x = self.agent.get_ball_pos()[0] - self.camera_bias
            time.sleep(0.5)
            # 当前约定：球偏左时倾向左脚，偏右时倾向右脚。
            if ball_x > 0:
                self.agent.kick(foot=0)
            else:
                self.agent.kick(foot=1)
            time.sleep(2)
            self.agent.move_head(inf, inf)
            self.logger.info(
                "===============================[KICK FSM] Kick executed successfully!========================="
            )
            self.logger.info(
                "===============================[KICK FSM] Kick executed successfully!========================="
            )
            self.logger.info(
                "===============================[KICK FSM] Kick executed successfully!========================="
            )
            self.finish()

    def stop(self):
        """停止移动。"""
        self.logger.debug("[KICK FSM] Stopping robot...")
        self.agent.stop(1)
        self.logger.debug("[KICK FSM] Robot stopped.")

    def adjust_angle(self):
        """调整身体朝向，让球和球门大致落在同一射门方向上。"""
        self.logger.debug("[ANGLE ADJUST] Starting angle adjustment...")

        self.logger.debug(
            f"[ANGLE ADJUST] Target angle: {self.ang_tar:.2f}°, Current yaw: {self.agent.get_self_yaw():.2f}°, Delta: {self.ang_delta:.2f}°"
        )

        ball_y_distance = (
            self.agent.get_ball_pos()[1] if not self.agent.get_ball_pos()[1] is None else 0.5
        )
        max_y_vel = self.agent.get_config().get("max_walk_vel_y")
        max_theta_vel = self.agent.get_config().get("max_walk_vel_theta")
        ratio = max_y_vel / max_theta_vel * self.moonwalk_vel_ratio

        if self.ang_delta > self.good_angle_threshold_degree:
            self.logger.debug(f"[ANGLE ADJUST] Rotating CCW (Δ={self.ang_delta:.2f}°)")
            self.agent.cmd_vel(
                0, -ball_y_distance * self.rotate_vel_theta / ratio, self.rotate_vel_theta
            )
        elif self.ang_delta < -self.good_angle_threshold_degree:
            self.logger.debug(f"[ANGLE ADJUST] Rotating CW (Δ={self.ang_delta:.2f}°)")
            self.agent.cmd_vel(
                0, ball_y_distance * self.rotate_vel_theta / ratio, -self.rotate_vel_theta
            )

    def good_angle(self):
        """判断当前朝向是否已经足够好。"""

        result = abs(self.ang_delta) < self.good_angle_threshold_degree

        self.logger.debug(
            f"[ANGLE CHECK] Angle delta: {abs(self.ang_delta):.2f}° (OK? {'Yes' if result else 'No'})"
        )
        return result

    def really_not_good_angle(self):
        """判断当前朝向是否已经差到需要回退到角度调整阶段。"""

        result = abs(self.ang_delta) >= self.really_bad_angle_threshold_degree

        self.logger.debug(
            f"[ANGLE CHECK] Large angle delta: {abs(self.ang_delta):.2f}° (Bad? {'Yes' if result else 'No'})"
        )
        return result

    def not_good_angle(self):
        """判断当前朝向是否还不够好。"""
        return not self.good_angle()

    def not_finished(self):
        """判断踢球对位是否尚未完成。"""
        return not (
            self.good_angle() or self.good_back_forth() or self.good_position_horizontally()
        )

    def adjust_horizontally(self):
        """左右微调，让球进入可出脚的横向窗口。"""
        self.logger.debug("\n[LR ADJUST] Starting lateral adjustment...")
        ball_x = self.agent.get_ball_pos()[0] - self.camera_bias
        center = (
            self.horizontal_position_lower_threshold_m + self.horizontal_position_upper_threshold_m
        ) / 2
        if abs(ball_x - center) < 0.2:
            y_vel = 0.5 * self.lateral_vel
        else:
            y_vel = self.lateral_vel

        if ball_x > center:
            self.logger.debug(f"[KICK LR ADJUST] Moving left (Ball X: {ball_x:.2f}mm)")
            self.agent.cmd_vel(0, -y_vel, 0)
        elif ball_x < center:
            self.logger.debug(f"[KICK LR ADJUST] Moving right (Ball X: {ball_x:.2f}mm)")
            self.agent.cmd_vel(0, y_vel, 0)

    def good_position_horizontally(self):
        """判断横向位置是否已经合适。"""
        ball_x = self.agent.get_ball_pos()[0]
        result = (
            self.horizontal_position_lower_threshold_m
            < ball_x
            < self.horizontal_position_upper_threshold_m
        )

        self.logger.debug(
            f"[LR CHECK] Ball X offset: {ball_x:.2f}mm (OK? {'Yes' if result else 'No'})"
        )
        return result

    def not_good_position_horizontally(self):
        """判断横向位置是否仍需调整。"""
        return not self.good_position_horizontally()

    def adjust_back_forth(self):
        """前后微调，让球位于脚前合适距离。"""
        self.logger.debug("\n[FB ADJUST] Starting forward adjustment...")
        ball_y = self.agent.get_ball_pos()[1]
        ball_x = self.agent.get_ball_pos()[0] - self.camera_bias

        horizontal_center = (
            self.horizontal_position_lower_threshold_m + self.horizontal_position_upper_threshold_m
        ) / 2

        y_vel = 0.0
        # if ball_x > horizontal_center:
        #     y_vel = - self.lateral_vel
        # else:
        #     y_vel = self.lateral_vel

        if ball_y > self.max_kick_y_distance_m:
            if ball_y > self.max_kick_y_distance_m + 0.1:
                self.logger.debug(f"[FB ADJUST] Moving forward (Distance: {ball_y:.2f}m)")
                self.agent.cmd_vel(self.forward_vel, y_vel, 0)
            else:
                self.agent.cmd_vel(self.forward_vel / 2, y_vel, 0)
        elif ball_y < self.min_kick_y_distance_m:
            self.logger.debug(f"[FB ADJUST] Moving backward (Distance: {ball_y:.2f}m)")
            self.agent.cmd_vel(-self.back_vel, y_vel, 0)

    def good_back_forth(self):
        """判断前后距离是否已经合适。"""
        ball_y = self.agent.get_ball_pos()[1]
        result = self.min_kick_y_distance_m < ball_y < self.max_kick_y_distance_m

        self.logger.debug(
            f"[FB CHECK] Ball Distance: {ball_y:.2f}m (OK? {'Yes' if result else 'No'})"
        )
        return result

    def not_good_back_forth(self):
        """判断前后距离是否仍需调整。"""
        return not self.good_back_forth()

    def read_params(self):
        """读取配置参数"""
        dribble_config = self._config.get("kick", {})
        self.good_angle_threshold_degree = dribble_config.get("good_angle_threshold_degree", 10)
        self.really_bad_angle_threshold_degree = dribble_config.get(
            "really_bad_angle_threshold_degree", 30
        )
        self.horizontal_position_upper_threshold_m = dribble_config.get(
            "horizontal_position_upper_threshold_m", 0.3
        )
        self.horizontal_position_lower_threshold_m = dribble_config.get(
            "horizontal_position_lower_threshold_m", 0.3
        )

        self.min_kick_y_distance_m = dribble_config.get("min_kick_y_distance_m", 0.12)
        self.max_kick_y_distance_m = dribble_config.get("max_kick_y_distance_m", 0.16)
        self.lost_ball_distance_threshold_m = dribble_config.get(
            "lost_ball_distance_threshold_m", 0.8
        )

        self.forward_vel = dribble_config.get("forward_vel", 0.3)
        self.back_vel = dribble_config.get("back_vel", 0.3)
        self.lateral_vel = dribble_config.get("lateral_vel", 0.05)
        self.rotate_vel_theta = dribble_config.get("rotate_vel_theta", 0.3)
        self.horizontal_adjust_forward_vel = dribble_config.get(
            "horizontal_adjust_forward_vel", 0.05
        )
        self.adjust_angle_vel_y = dribble_config.get("adjust_angle_vel_y", 0.05)
        self.blind_dribble_thres_percent = dribble_config.get("blind_dribble_thres_percent", 1.0)
        self.camera_bias = dribble_config.get("camera_bias", 0.035)
        self.blind_kick_thres_percent = dribble_config.get("blind_kick_thres_percent", 1.0)
        self.moonwalk_vel_ratio = dribble_config.get("moonwalk_vel_ratio", 0.3)

    # 配置文件（JSON格式）
    """
    "kick": {
        "good_angle_threshold_degree": 10,
        "really_bad_angle_threshold_degree": 30,
        "horizontal_position_threshold_mm": 30,
        "horizontal_adjust_threshold_degree": 5,
        "min_kick_y_distance_m": 0.3,
        "max_kick_y_distance_m": 0.35,
        "lost_ball_distance_threshold_m": 0.8,
        "forward_vel": 0.3,
        "lateral_vel": 0.05,
        "rotate_vel_theta": 0.3,
        "horizontal_adjust_forward_vel": 0.05,
        "adjust_angle_vel_y": 0.05
    }
    """
