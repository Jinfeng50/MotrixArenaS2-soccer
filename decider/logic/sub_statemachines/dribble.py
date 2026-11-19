import math
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque


class DribbleStateMachine:
    """
    连续 P 控制带球器。

    与离散状态机不同，这里每一拍都直接根据球的位置和目标朝向
    计算一个连续速度控制量。
    """

    def __init__(self, agent):
        """
        初始化 P 控制带球器。
        """
        self.agent = agent
        self.logger = self.agent.get_logger().get_child("dribble_p_control")
        self._config = self.agent.get_config()
        self.read_params()
        self.logger.info("[DRIBBLE P-CONTROL] Initialized.")

    def read_params(self):
        """从配置中读取带球控制参数。"""
        dribble_cfg = self._config.get("dribble", {})

        # 三个方向的比例控制系数。
        self.kp_x = dribble_cfg.get("kp_x", 1.5)
        self.kp_y = dribble_cfg.get("kp_y", 1.5)
        self.kp_theta = dribble_cfg.get("kp_theta", 1.0)

        # 两种期望距离：
        # 1. dist_setup：尚未对正目标方向时，先站到球后方较安全的位置
        # 2. dist_dribble：已经对正后，允许更激进地顶球向前
        self.dist_setup = dribble_cfg.get("dist_setup", 0.45)
        self.dist_dribble = dribble_cfg.get("dist_dribble", -0.1)

        # 角度阈值：决定当前应处于“对位”还是“推进”模式。
        self.align_threshold_deg = dribble_cfg.get("align_threshold_deg", 15.0)
        self.align_threshold_rad = math.radians(self.align_threshold_deg)

        # 速度上限与最小推进速度。
        self.max_vel_linear = dribble_cfg.get("max_vel_linear", 1.0)
        self.min_push_vel = dribble_cfg.get("min_push_vel", 0.5)
        self.orbit_efficiency = dribble_cfg.get("orbit_efficiency", 0.8)
        self.stop_threshold_dist = 0.05

        # 调试记录：把误差和控制量画图，便于线下调参。
        self.history_len = 500
        self.history = {
            "time": deque(maxlen=self.history_len),
            "e_x": deque(maxlen=self.history_len),
            "e_y": deque(maxlen=self.history_len),
            "e_th": deque(maxlen=self.history_len),
            "v_x": deque(maxlen=self.history_len),
            "v_y": deque(maxlen=self.history_len),
            "v_th": deque(maxlen=self.history_len),
            "mode": deque(maxlen=self.history_len),
        }
        self.plot_interval = 10
        self.step_count = 0
        self.plot_save_path = os.path.join(
            os.path.dirname(__file__), "../../logs/dribble_debug.png"
        )
        os.makedirs(os.path.dirname(self.plot_save_path), exist_ok=True)

    def run(self, aim_yaw=None, _kick_when_dribble=False):
        """
        执行一拍连续带球控制。

        核心思路：
        1. 先确定目标朝向（通常是朝球门）。
        2. 根据当前是否对正，决定使用 SETUP 还是 PUSH 模式。
        3. 把球理想上应位于身体前方的目标位置转成误差，再映射成速度命令。
        """
        # 1. 看不到球时立即停车。
        if not self.agent.get_if_ball():
            self.logger.warning("[DRIBBLE] No ball seen. Stopping.")
            self.agent.cmd_vel(0, 0, 0)
            return

        # 2. 读取输入：球位置和当前朝向。
        ball_pos_vis = self.agent.get_ball_pos()
        b_forward, b_left = ball_pos_vis[0], ball_pos_vis[1]

        b_x = b_forward  # Forward
        b_y = b_left  # Left

        curr_yaw = self.agent.get_self_yaw()

        # 目标朝向优先级：
        # 1. run() 显式传入
        # 2. 外部命令黑板指定
        # 3. 默认朝对方球门中心
        target_yaw = self._get_target_yaw(aim_yaw)

        # 3. 姿态误差 alpha：表示“机器人当前朝向”和“希望推进方向”之间的夹角。
        alpha = self.agent.angle_normalize(target_yaw - curr_yaw)

        self.logger.debug(
            f"[DRIBBLE] Ball: ({b_x:.2f}, {b_y:.2f}), Alpha: {math.degrees(alpha):.1f} deg"
        )

        # 4. 若已对正，就进入 PUSH 模式；否则先进入 SETUP 模式重新站位。
        if abs(alpha) < self.align_threshold_rad:
            target_dist = self.dist_dribble
            mode = "PUSH"
        else:
            target_dist = self.dist_setup
            mode = "SETUP"

        # 5. 根据目标朝向构造一个“理想球位”。
        # 直观理解：
        # - PUSH 模式下，希望球更贴近身体前方，便于持续顶球。
        # - SETUP 模式下，希望自己先站到球后面，避免斜着把球碰丢。

        vec_target_local_x = math.cos(alpha)
        vec_target_local_y = math.sin(alpha)

        p_des_x = b_x - target_dist * vec_target_local_x
        p_des_y = b_y - target_dist * vec_target_local_y

        # 误差向量就是理想球位相对当前机体原点的偏差。
        e_x = p_des_x
        e_y = p_des_y
        e_theta = alpha

        self.logger.debug(
            f"[DRIBBLE] Mode: {mode}, T_Dist: {target_dist}, Errors: x={e_x:.2f}, y={e_y:.2f}, th={e_theta:.2f}"
        )

        v_x = self.kp_x * e_x
        v_y = self.kp_y * e_y
        v_theta = self.kp_theta * e_theta

        # PUSH 模式下始终保留最小向前速度，避免靠近球时来回抖动却不真正推进。
        if mode == "PUSH":
            if v_x < self.min_push_vel:
                v_x = self.min_push_vel

        # SETUP 模式下不应一味向前顶，而应优先侧移 / 转向，把自己摆到球后方。
        if mode == "SETUP":
            pass

            # 一个简化启发式：角度偏差越大，越压低前进量，鼓励绕球对位。
            damp_factor = max(0.0, 1.0 - (abs(alpha) / math.pi))
            v_x *= damp_factor

        # 6. 对线速度做合速度限幅。
        v_linear_mag = math.hypot(v_x, v_y)
        if v_linear_mag > self.max_vel_linear:
            scale = self.max_vel_linear / v_linear_mag
            v_x *= scale
            v_y *= scale

        # 统一输出关键信息，便于看日志判断带球模式切换是否合理。
        log_msg = (
            f"[DRIBBLE] Mode: {mode[0]} | "
            f"Alph: {math.degrees(alpha):.0f} | "
            f"T_Yaw: {math.degrees(target_yaw):.0f} | "
            f"Bal: ({b_x:.2f},{b_y:.2f}) | "
            f"Cmd: {v_x:.2f}, {v_y:.2f}, {v_theta:.2f}"
        )
        self.logger.info(log_msg)

        # 7. 下发速度命令。
        self.agent.cmd_vel(v_x, v_y, v_theta)
        self.agent.move_head(math.inf, math.inf)

        # 8. 记录调试数据。
        self._record_data(e_x, e_y, e_theta, v_x, v_y, v_theta, mode)

    def _record_data(self, ex, ey, eth, vx, vy, vth, mode):
        self.step_count += 1
        self.history["time"].append(self.step_count)
        self.history["e_x"].append(ex)
        self.history["e_y"].append(ey)
        self.history["e_th"].append(math.degrees(eth))
        self.history["v_x"].append(vx)
        self.history["v_y"].append(vy)
        self.history["v_th"].append(vth)
        self.history["mode"].append(1 if mode == "PUSH" else 0)

        if self.step_count % self.plot_interval == 0:
            self._save_plot()

    def _save_plot(self):
        try:
            plt.figure(figsize=(10, 8))

            # Subplot 1: Errors
            plt.subplot(3, 1, 1)
            plt.plot(self.history["time"], self.history["e_x"], label="Err X (m)")
            plt.plot(self.history["time"], self.history["e_y"], label="Err Y (m)")
            plt.title("Position Errors (Control Frame)")
            plt.grid(True)
            plt.legend()

            # Subplot 2: Angular Error
            plt.subplot(3, 1, 2)
            plt.plot(
                self.history["time"], self.history["e_th"], label="Err Theta (deg)", color="orange"
            )
            plt.plot(
                self.history["time"],
                self.history["mode"],
                label="Mode (1=Push)",
                color="green",
                alpha=0.3,
            )
            plt.title("Angular Error & Mode")
            plt.grid(True)
            plt.legend()

            # Subplot 3: Commands
            plt.subplot(3, 1, 3)
            plt.plot(self.history["time"], self.history["v_x"], label="Cmd Vx")
            plt.plot(self.history["time"], self.history["v_y"], label="Cmd Vy")
            plt.title("Velocity Commands")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig(self.plot_save_path)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Plotting failed: {e}")

    def _get_target_yaw(self, override_yaw=None):
        """确定本拍带球目标朝向。"""
        # 1. 优先使用 run() 直接传入的目标朝向。
        if override_yaw is not None:
            return override_yaw

        # 2. 其次使用外部命令黑板中的 aim_yaw。
        cmd_yaw = self.agent.get_command().get("data", {}).get("aim_yaw", None)
        if cmd_yaw is not None:
            return cmd_yaw

        # 3. 最后默认朝对方球门中心推进。

        try:
            field_len = self._config.get("field_size", {}).get(self.agent.league, [14.0, 9.0])[0]

            goal_x = field_len / 2.0
            my_pos = self.agent.get_self_pos()

            if my_pos is not None and len(my_pos) == 2 and my_pos[0] is not None:
                goal_angle = math.atan2(0 - my_pos[1], goal_x - my_pos[0])
                return goal_angle
        except Exception as e:
            self.logger.warning(f"Error calcing target yaw: {e}")
            pass

        return 0.0
