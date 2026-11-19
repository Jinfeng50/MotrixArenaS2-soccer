# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# 决策端主入口。
# 这个文件同时支持两种运行方式：
# 1. ROS2 真机模式：通过 ROS2 话题与机器人其他模块交互。
# 2. 仿真模式：通过 ZMQ 与仿真器做命令-状态锁步同步。

import sys, os, math, signal, time, traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List

# 条件导入 ROS2，保证没有 ROS2 时也能跑纯仿真模式。
ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node

    ROS_AVAILABLE = True
except ImportError:
    # 未安装 ROS2 时，仅保留仿真模式能力。
    Node = object  # 退化基类，占位用

# 与感知 / 动作相关的接口层。
import interfaces.action
import interfaces.vision
import configuration

# 用户自定义比赛逻辑入口。
import user_entry

MAX_SIM_ROBOTS_PER_TEAM = 7


class Agent(Node):
    """
    真机 / ROS2 模式下的主 Agent。

    这个类提供统一的感知、动作和配置接口，
    让上层比赛状态机不用关心底层是 ROS2 还是真机通信。
    """

    def __init__(self, args=None):
        """初始化 ROS2 Agent，并搭建比赛策略运行骨架。"""
        super().__init__("decider")
        self.logger = self.get_logger()
        self.logger.info("[Core] Initializing the core")
        self._config = configuration.load_config()
        self._action = interfaces.action.Action(self)
        self._vision = interfaces.vision.Vision(self)
        self.logger.info("[Core] Core initialized. Calling user's init()")

        # user_entry.init() 是策略真正接入系统的地方。
        # 所有状态机和比赛逻辑都会在这里挂到 agent 上。
        user_entry.init(self)

        # 创建定时器，替代ROS 1中的while循环
        timer_period = 1.0 / 10  # 默认10Hz
        self.timer = self.create_timer(timer_period, self._timer_callback)
        self.logger.info("[Core] ALl initializing finished. ")

    def _timer_callback(self):
        """ROS2 定时器回调：每个周期执行一次用户比赛逻辑。"""
        try:
            user_entry.loop(self)
        except Exception as e:
            self.logger.error(f"[Core] Error in timer callback: {e}")
            self.logger.error(f"[Core] Core exited !")
            self.stop()
            self.logger.error(f"[Core] The trackback infomation is shown below:")
            traceback.print_exc()
            exit(-1)

    def cmd_vel(self, vel_x: float, vel_y: float, vel_theta: float) -> None:
        """发送速度指令，并按配置把控制量映射到实际速度范围。"""
        # 上层状态机一般输出的是归一化控制量。
        # 这里统一做三件事：
        # 1. 按配置缩放到机器人允许的线速度 / 角速度范围。
        # 2. 对很小但非零的速度补一个起步量，避免“想动却迈不出去”。
        # 3. 最后裁剪到 [-1, 1]，防止下层接口收到越界值。
        vel_x *= self._config.get("max_walk_vel_x", 0.25)
        vel_y *= self._config.get("max_walk_vel_y", 0.1)
        vel_theta *= self._config.get("max_walk_vel_theta", 0.5)
        if vel_x > 0.001:
            vel_x += self._config.get("min_walk_vel_x", 0.2)
        elif vel_x < -0.001:
            vel_x -= self._config.get("min_walk_vel_x", 0.2)
        if vel_y > 0.001:
            vel_y += self._config.get("min_walk_vel_y", 0.2)
        elif vel_y < -0.001:
            vel_y -= self._config.get("min_walk_vel_y", 0.2)
        if vel_theta > 0.001:
            vel_theta += self._config.get("min_walk_vel_theta", 0.3)
        elif vel_theta < -0.001:
            vel_theta -= self._config.get("min_walk_vel_thetea", 0.3)
        vel_x = float(np.clip(vel_x, -1.0, 1.0))
        vel_y = float(np.clip(vel_y, -1.0, 1.0))
        vel_theta = float(np.clip(vel_theta, -1.0, 1.0))
        self._action.cmd_vel(vel_x, vel_y, vel_theta)

    def move_head(self, pitch: float, yaw: float) -> None:
        """设置头部俯仰与偏航角。"""
        self._action._move_head(pitch, yaw)

    def stop(self, sleep_time: float = 0) -> None:
        """立即停车；必要时额外等待一小段时间让动作稳定。"""
        self._action.cmd_vel(0.0, 0.0, 0.0)
        time.sleep(sleep_time)

    def kick(self, foot=0, death=0) -> None:
        """执行踢球动作。"""
        self._action.do_kick(foot, death)

    def save_ball(self, direction=1):
        """执行扑救动作。"""
        self._action.save_ball(direction)

    def angle_normalize(self, angle: float) -> float:
        """把角度归一化到 (-pi, pi]，避免角度跳变影响控制。"""
        if angle is None:
            return None
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        self.get_logger().debug(f"Normalized angle: {angle}")
        return angle

    def relocate(self, x: float = 0, y: float = 0, theta: float = 0):
        """把当前位置重置为新的参考位姿。"""
        return self._vision.relocate(x, y, theta)

    def get_objects(self) -> list:
        # A list of dict:
        #    {
        #        'label': label,
        #        'relative_pos': relative_pos,
        #        'absolute_pos': absolute_coord,
        #        'distance': distance,
        #        'confidence': obj.confidence,
        #        'bounding_box_center': curr_coord / 1000,
        #        'bound_left_low': np.array(obj.bound_left_low[:2]) if obj.bound_left_low else None,
        #        'bound_right_low': np.array(obj.bound_right_low[:2]) if obj.bound_right_low else None,
        #        'timestamp': time.time()
        #    }
        return self._vision.get_objects()

    def get_self_pos(self) -> np.ndarray:
        """返回机器人在地图坐标系下的位置。"""
        return self._vision.self_pos

    def get_self_yaw(self) -> float:
        """返回机器人在地图坐标系下的朝向角。"""
        return self._vision.self_yaw

    def get_ball_pos(self) -> List[Optional[float]]:
        """返回球在机器人坐标系下的位置；未看到球时返回 [None, None]。"""
        if self.get_if_ball():
            return self._vision.get_ball_pos()
        else:
            return [None, None]

    def get_ball_angle(self) -> Optional[float]:
        """计算球相对机器人正前方的夹角。

        当前统一采用：
        - X：前方
        - Y：左方

        因此角度定义为 atan2(球的左向分量, 球的前向分量)。
        结果语义：
        - 正值：球在左边
        - 负值：球在右边
        """
        ball_pos = self.get_ball_pos()
        ball_x = ball_pos[0]  # Forward
        ball_y = ball_pos[1]  # Left

        if (
            ball_pos is not None
            and ball_x is not None
            and ball_y is not None
            and self.get_if_ball()
        ):
            if ball_x == 0 and ball_y == 0:
                return None
            else:
                angle_rad = math.atan2(ball_y, ball_x)
                return angle_rad
        else:
            return None

    def get_ball_pos_in_vis(self) -> np.ndarray:
        """返回球在视觉图像坐标中的位置。"""
        return self._vision.get_ball_pos_in_vis()

    def get_ball_pos_in_map(self) -> np.ndarray:
        """返回球在全局地图坐标系中的位置。"""
        if not self.get_if_ball():
            self.get_logger().warning("Ball not detected, returning None")
            return None
        return self._vision.get_ball_pos_in_map()

    def get_if_ball(self) -> bool:
        """返回当前是否看到球。"""
        return self._vision.get_if_ball()

    def get_if_close_to_ball(self) -> bool:
        """返回当前是否已经接近球。"""
        if self.get_if_ball():
            return self.get_ball_distance() < self._config.get("close_to_ball_threshold", 0.5)
        else:
            return False

    def get_ball_distance(self) -> float:
        """返回到球的距离；未看到球时返回一个很大的默认值。"""
        if self.get_if_ball():
            return self._vision.ball_distance
        else:
            return 1e6

    def get_neck(self) -> float:
        """返回颈部角度。"""
        return self._vision.neck

    def get_head(self) -> float:
        """返回头部角度。"""
        return self._vision.head

    def get_ball_history(self):
        """返回球的历史观测轨迹。"""
        return self._vision.get_ball_history()

    def get_config(self) -> Dict:
        """返回当前配置字典。"""
        return self._config


class SimAgent:
    """
    仿真模式下的 Agent。

    它不依赖 ROS2，而是通过 ZMQ 与仿真器做请求-响应同步。
    对上层策略来说，它尽量模拟 Agent 的接口，
    这样同一套状态机代码可以尽量少改地复用于仿真和真机。
    """

    class _ROS2CompatibleLogger:
        """用 Python logging 模拟 ROS2 logger 接口。"""

        def __init__(self, name):
            self._logger = logging.getLogger(name)

        def get_child(self, suffix):
            """模拟 ROS2 的 get_child，生成层级 logger。"""
            return SimAgent._ROS2CompatibleLogger(f"{self._logger.name}.{suffix}")

        def info(self, msg):
            self._logger.info(msg)

        def debug(self, msg):
            self._logger.debug(msg)

        def warning(self, msg):
            self._logger.warning(msg)

        def error(self, msg):
            self._logger.error(msg)

        def fatal(self, msg):
            self._logger.critical(msg)


def _configure_sim_logging():
    log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    enable_file_log = os.environ.get("MOS_BRAIN_SIM_FILE_LOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    log_level_name = os.environ.get("MOS_BRAIN_SIM_LOG_LEVEL", "WARNING").strip().upper()
    log_level = getattr(logging, log_level_name, logging.WARNING)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)

    handlers = [console_handler]
    if enable_file_log:
        try:
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(
                f"logs/sim_decider_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler.setFormatter(log_fmt)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Failed to setup file logging: {e}")

    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    def __init__(self, args=None, ip="127.0.0.1", port="5555", sim_hz=None):
        _configure_sim_logging()

        self.logger = SimAgent._ROS2CompatibleLogger("SimDecider")
        self.logger.info(
            "[SimCore] Initializing SimAgent in ROS-Free Mode (Logging to Console + File)"
        )

        self.is_simulation = True
        self._config = configuration.load_config()
        self.id = self._config.get("id", 0)
        self.red_count = None
        self.team_size = int(os.environ.get("MOS_BRAIN_TEAM_SIZE", "0") or 0)
        self.league = self._config.get("league", "S")  # 显式记录场地规格，缺省按 S 处理。
        # 仿真控制频率（Hz）。
        # 小于等于 0 表示不主动 sleep，让循环按仿真最大速度推进。
        self.sim_hz = self._config.get("sim_hz", 50.0) if sim_hz is None else sim_hz

        # 若命令行提供了参数，则覆盖配置文件默认值。
        if args:
            if hasattr(args, "id") and args.id is not None:
                self.id = args.id
                self._config["id"] = self.id  # 同步回配置，保证 Vision 读取到的是正确本地编号。

            # 队伍颜色优先使用命令行，其次才回退配置文件。
            self.color = self._config.get("color", "red")
            if hasattr(args, "color") and args.color is not None:
                self.color = args.color

            if hasattr(args, "sim_hz") and args.sim_hz is not None:
                self.sim_hz = args.sim_hz

            if hasattr(args, "red_count") and args.red_count is not None:
                self.red_count = int(args.red_count)

            # 根据红蓝队身份，把“队内编号”转换成仿真器里的“全局编号”。
            if self.color:
                try:
                    red_count = MAX_SIM_ROBOTS_PER_TEAM
                    if self.red_count is not None and self.red_count != MAX_SIM_ROBOTS_PER_TEAM:
                        self.logger.warning(
                            f"[SimCore] Ignoring --red-count={self.red_count} for blue offset; simulator uses fixed team slots={MAX_SIM_ROBOTS_PER_TEAM}"
                        )

                    if self.color == "blue":
                        # 仿真器里的机器人编号是一个扁平空间：
                        # 仿真器的蓝队全局 id 按固定队伍槽位偏移，不按当前上场人数偏移。
                        self.logger.info(
                            f"[SimCore] Team Blue specified. Adding offset {red_count} to ID {self.id}"
                        )
                        self.id += red_count
                    elif self.color == "red":
                        self.logger.info(f"[SimCore] Team Red specified. ID {self.id} unchanged")

                except Exception as e:
                    self.logger.warning(
                        f"[SimCore] Failed to load match config for team offset: {e}. Using raw ID."
                    )

        try:
            self.sim_hz = float(self.sim_hz)
        except (TypeError, ValueError):
            self.logger.warning(f"[SimCore] Invalid sim_hz={self.sim_hz}, fallback to 50.0")
            self.sim_hz = 50.0

        if self.sim_hz > 0:
            self.logger.info(f"[SimCore] 控制频率限制为 {self.sim_hz:.2f} Hz")
        else:
            self.logger.info("[SimCore] 控制频率不设上限（按锁步最大速度运行）")

        self.logger.info(f"[SimCore] Final Robot ID: {self.id}")

        # 初始化仿真通信客户端。
        from interfaces.sim_client import SimClient

        self.client = SimClient(ip=ip, port=port)

        # 初始化动作 / 感知接口。
        # 它们会根据 is_simulation 标志切换到仿真实现。
        self._action = interfaces.action.Action(self)
        self._vision = interfaces.vision.Vision(self)

        # 裁判状态与队友通信缓存也挂在这里，供 user_entry 直接读取。
        from interfaces.gamecontroller import GameController
        from interfaces.communication import Communication

        self.gamecontroller = GameController(self)
        self.communication = Communication(self)

        self.logger.info("[SimCore] Core initialized. Calling user's init()")
        user_entry.init(self)

        # 当前待发送给仿真器的速度命令。
        self.current_cmd = [0.0, 0.0, 0.0]
        self.sim_now = 0.0

    def get_logger(self):
        return self.logger

    def get_clock(self):
        # 提供一个最小可用的“仿 ROS2 时钟”接口，方便复用旧代码。
        class MockClock:
            def now(self):
                class MockTime:
                    def to_msg(self):
                        return time.time()

                return MockTime()

        return MockClock()

    def run(self):
        self.logger.info("[SimCore] Starting Loop...")
        tick_period = (1.0 / self.sim_hz) if self.sim_hz > 0 else None
        while True:
            try:
                loop_start = time.perf_counter()
                # 1. 先执行用户策略。
                # 这里读到的是“上一拍同步后”的状态缓存。
                user_entry.loop(self)

                # 2. 把当前速度命令发给仿真器，并同步拿回最新世界状态。
                state = self.client.communicate(self.current_cmd, robot_id=self.id)

                if state:
                    self.sim_now = float(state.get("sim_timestamp", time.time()))
                    # 3. 把仿真状态拆给感知、裁判状态和通信模块。
                    sim_state = state.get("state", {})
                    self._vision.update_from_sim_state(sim_state)
                    self.gamecontroller.update(sim_state.get("gamecontroller", {}))
                    self.communication.update(sim_state.get("communication", {}))

                if tick_period is not None:
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = tick_period - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"[SimCore] Error in loop: {e}")
                traceback.print_exc()
                break

        self.stop()

    def cmd_vel(self, vel_x: float, vel_y: float, vel_theta: float) -> None:
        """设置仿真机器人的速度命令，并缓存到下一次同步发送。"""
        # 这里复用与真机模式相同的缩放语义，保证上层状态机不用区分后端。
        vel_x *= self._config.get("max_walk_vel_x", 0.25)
        vel_y *= self._config.get("max_walk_vel_y", 0.1)
        vel_theta *= self._config.get("max_walk_vel_theta", 0.5)
        vel_x = float(np.clip(vel_x, -1.0, 1.0))
        vel_y = float(np.clip(vel_y, -1.0, 1.0))
        vel_theta = float(np.clip(vel_theta, -1.0, 1.0))
        self._action.cmd_vel(vel_x, vel_y, vel_theta)

    def move_head(self, pitch: float, yaw: float) -> None:
        self._action._move_head(pitch, yaw)

    def stop(self, sleep_time: float = 0) -> None:
        self.current_cmd = [0.0, 0.0, 0.0]
        self._action.cmd_vel(0.0, 0.0, 0.0)

        # 停车时额外强制同步一次 0 速度，避免仿真器还保持旧命令。
        try:
            if hasattr(self, "client"):
                self.client.force_reset()
                self.client.communicate(self.current_cmd, robot_id=self.id)
        except Exception as e:
            # Ignore errors during shutdown
            pass

        time.sleep(sleep_time)

    def kick(self, foot=0, death=0) -> None:
        self._action.do_kick(foot, death)

    def save_ball(self, direction=1):
        self._action.save_ball(direction)

    def angle_normalize(self, angle: float) -> float:
        if angle is None:
            return None
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # 以下感知接口全部转发给 vision 模块，对上层保持与真机 Agent 一致。
    def relocate(self, x=0, y=0, theta=0):
        return self._vision.relocate(x, y, theta)

    def get_objects(self):
        return self._vision.get_objects()

    def get_self_pos(self):
        return self._vision.self_pos

    def get_self_yaw(self):
        return self._vision.self_yaw

    def get_ball_pos(self):
        return self._vision.get_ball_pos() if self.get_if_ball() else [None, None]

    def get_ball_angle(self):
        """计算球相对机器人正前方的角度。"""
        ball_pos = self.get_ball_pos()
        if ball_pos[0] is None:
            return None
        if ball_pos[0] == 0 and ball_pos[1] == 0:
            return None
        return math.atan2(ball_pos[1], ball_pos[0])

    def get_ball_pos_in_vis(self):
        return self._vision.get_ball_pos_in_vis()

    def get_ball_pos_in_map(self):
        return self._vision.get_ball_pos_in_map()

    def get_if_ball(self):
        return self._vision.get_if_ball()

    def get_if_close_to_ball(self):
        return (
            (self.get_ball_distance() < self._config.get("close_to_ball_threshold", 0.5))
            if self.get_if_ball()
            else False
        )

    def get_ball_distance(self):
        return self._vision.ball_distance if self.get_if_ball() else 1e6

    def get_neck(self):
        return self._vision.neck

    def get_head(self):
        return self._vision.head

    def get_ball_history(self):
        return self._vision.get_ball_history()

    def get_config(self):
        return self._config

    def get_command(self):
        """
        给旧状态机提供一个命令黑板接口。

        注意：当前返回的是 `params` 字段，而仓库里有一些旧状态机会读取 `data`。
        如果那部分代码没做兼容，就会回退到默认参数。
        """
        params = getattr(self, "_command_params", {})
        return {
            "command": getattr(self, "_command", "stop"),
            "params": params,
            "data": params,
        }

    def set_command(self, command: str, params: dict = None):
        """设置当前高层命令与附带参数。"""
        self._command = command
        self._command_params = params or {}

    def get_obstacle_avoidance_velocity(self):
        """
        障碍规避占位实现。

        当前仿真模式下默认不做实际障碍规避，
        调用方据此回退到普通追球 / 回场控制。
        """
        return None, None, None


class TeamSimAgent(SimAgent):
    """单进程批量仿真模式下的机器人代理。"""

    MAX_ROBOTS_PER_TEAM = 7

    def __init__(
        self,
        shared_client,
        *,
        local_id: int,
        color: str,
        team_size: int,
        sim_hz: float | None = None,
    ):
        self.logger = SimAgent._ROS2CompatibleLogger(f"SimDecider.{color}.{local_id}")
        self.is_simulation = True
        self._config = configuration.load_config()
        self._config["id"] = int(local_id)
        self._config["color"] = str(color)
        self.color = str(color)
        self.local_id = int(local_id)
        self.team_size = int(team_size)
        self.red_count = self.MAX_ROBOTS_PER_TEAM
        self.id = self.local_id + (self.MAX_ROBOTS_PER_TEAM if self.color == "blue" else 0)
        self.league = self._config.get("league", "S")
        self.sim_hz = self._config.get("sim_hz", 50.0) if sim_hz is None else sim_hz
        self.client = shared_client

        self._action = interfaces.action.Action(self)
        self._vision = interfaces.vision.Vision(self)

        from interfaces.gamecontroller import GameController
        from interfaces.communication import Communication

        self.gamecontroller = GameController(self)
        self.communication = Communication(self)

        self.current_cmd = [0.0, 0.0, 0.0]
        self.sim_now = 0.0
        user_entry.init(self)

    def run(self):
        raise RuntimeError("TeamSimAgent does not support standalone run(); use TeamSimRunner.")

    def get_logger(self):
        return self.logger

    def get_clock(self):
        class MockClock:
            def now(self):
                class MockTime:
                    def to_msg(self):
                        return time.time()

                return MockTime()

        return MockClock()

    def get_config(self):
        return self._config

    def cmd_vel(self, vel_x: float, vel_y: float, vel_theta: float) -> None:
        vel_x *= self._config.get("max_walk_vel_x", 0.25)
        vel_y *= self._config.get("max_walk_vel_y", 0.1)
        vel_theta *= self._config.get("max_walk_vel_theta", 0.5)
        vel_x = float(np.clip(vel_x, -1.0, 1.0))
        vel_y = float(np.clip(vel_y, -1.0, 1.0))
        vel_theta = float(np.clip(vel_theta, -1.0, 1.0))
        self._action.cmd_vel(vel_x, vel_y, vel_theta)

    def move_head(self, pitch: float, yaw: float) -> None:
        self._action._move_head(pitch, yaw)

    def kick(self, foot=0, death=0) -> None:
        self._action.do_kick(foot, death)

    def save_ball(self, direction=1):
        self._action.save_ball(direction)

    def angle_normalize(self, angle: float) -> float:
        if angle is None:
            return None
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def relocate(self, x=0, y=0, theta=0):
        return self._vision.relocate(x, y, theta)

    def get_objects(self):
        return self._vision.get_objects()

    def get_self_pos(self):
        return self._vision.self_pos

    def get_self_yaw(self):
        return self._vision.self_yaw

    def get_ball_pos(self):
        return self._vision.get_ball_pos() if self.get_if_ball() else [None, None]

    def get_ball_angle(self):
        ball_pos = self.get_ball_pos()
        if ball_pos[0] is None:
            return None
        if ball_pos[0] == 0 and ball_pos[1] == 0:
            return None
        return math.atan2(ball_pos[1], ball_pos[0])

    def get_ball_pos_in_vis(self):
        return self._vision.get_ball_pos_in_vis()

    def get_ball_pos_in_map(self):
        return self._vision.get_ball_pos_in_map()

    def get_if_ball(self):
        return self._vision.get_if_ball()

    def get_if_close_to_ball(self):
        return (
            (self.get_ball_distance() < self._config.get("close_to_ball_threshold", 0.5))
            if self.get_if_ball()
            else False
        )

    def get_ball_distance(self):
        return self._vision.ball_distance if self.get_if_ball() else 1e6

    def get_neck(self):
        return self._vision.neck

    def get_head(self):
        return self._vision.head

    def get_ball_history(self):
        return self._vision.get_ball_history()

    def get_command(self):
        params = getattr(self, "_command_params", {})
        return {
            "command": getattr(self, "_command", "stop"),
            "params": params,
            "data": params,
        }

    def set_command(self, command: str, params: dict = None):
        self._command = command
        self._command_params = params or {}

    def get_obstacle_avoidance_velocity(self):
        return None, None, None

    def stop(self, sleep_time: float = 0) -> None:
        self.current_cmd = [0.0, 0.0, 0.0]
        self._action.cmd_vel(0.0, 0.0, 0.0)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def sync_from_sim_state(self, sim_state: dict):
        self._vision.update_from_sim_state(sim_state)
        self.gamecontroller.update(sim_state.get("gamecontroller", {}))
        self.communication.update(sim_state.get("communication", {}))


class TeamSimRunner:
    """用单进程统一驱动整队机器人，避免多进程串行 REQ/REP 带来的性能瓶颈。"""

    def __init__(self, args, ip="127.0.0.1", port="5555", sim_hz=None):
        _configure_sim_logging()
        self.logger = SimAgent._ROS2CompatibleLogger("SimTeamBatch")
        self.client = None
        self.sim_hz = sim_hz
        try:
            self.sim_hz = float(self.sim_hz) if self.sim_hz is not None else None
        except (TypeError, ValueError):
            self.sim_hz = None

        self.red_team_size = max(0, int(getattr(args, "red_team_size", 0) or 0))
        self.blue_team_size = max(0, int(getattr(args, "blue_team_size", 0) or 0))
        if self.red_team_size <= 0 and self.blue_team_size <= 0:
            raise ValueError("Team batch mode requires --red-team-size and/or --blue-team-size > 0")

        from interfaces.sim_client import SimClient

        self.client = SimClient(ip=ip, port=port)
        self.agents = []
        for i in range(self.red_team_size):
            self.agents.append(
                TeamSimAgent(
                    self.client,
                    local_id=i,
                    color="red",
                    team_size=self.red_team_size,
                    sim_hz=self.sim_hz,
                )
            )
        for i in range(self.blue_team_size):
            self.agents.append(
                TeamSimAgent(
                    self.client,
                    local_id=i,
                    color="blue",
                    team_size=self.blue_team_size,
                    sim_hz=self.sim_hz,
                )
            )

        if self.sim_hz is None and self.agents:
            self.sim_hz = self.agents[0].sim_hz

        self.logger.warning(
            f"[SimTeamBatch] batch mode enabled: red={self.red_team_size}, blue={self.blue_team_size}, sim_hz={self.sim_hz}"
        )

    def run(self):
        tick_period = (1.0 / self.sim_hz) if self.sim_hz and self.sim_hz > 0 else None
        while True:
            try:
                loop_start = time.perf_counter()
                for agent in self.agents:
                    user_entry.loop(agent)

                now = time.time()
                commands = [
                    {
                        "id": int(agent.id),
                        "cmd": list(agent.current_cmd),
                        "timestamp": now,
                        "source": "team_batch",
                    }
                    for agent in self.agents
                ]
                state = self.client.communicate_batch(commands)
                if state:
                    sim_ts = float(state.get("sim_timestamp", time.time()))
                    sim_state = state.get("state", {})
                    for agent in self.agents:
                        agent.sim_now = sim_ts
                        agent.sync_from_sim_state(sim_state)

                if tick_period is not None:
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = tick_period - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"[SimTeamBatch] Error in loop: {e}")
                traceback.print_exc()
                break

        self.stop()

    def stop(self):
        zero_commands = []
        for agent in self.agents:
            agent.stop()
            zero_commands.append(
                {
                    "id": int(agent.id),
                    "cmd": [0.0, 0.0, 0.0],
                    "timestamp": time.time(),
                    "source": "team_batch_stop",
                }
            )
        try:
            if self.client is not None:
                self.client.force_reset()
                self.client.communicate_batch(zero_commands)
                self.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true", help="Run in Sim Mode (ZMQ, No ROS)")
    parser.add_argument("--ip", default="127.0.0.1", help="Sim IP")
    parser.add_argument("--port", default="5555", help="Sim Port")
    parser.add_argument("--id", type=int, default=None, help="Robot ID (overrides config)")
    parser.add_argument(
        "--color",
        type=str,
        default=None,
        choices=["red", "blue"],
        help="Team color (red/blue). If set, --id is interpreted as player number within team.",
    )
    parser.add_argument(
        "--red-count",
        dest="red_count",
        type=int,
        default=None,
        help="Number of red-team robots in the target simulation.",
    )
    parser.add_argument(
        "--sim-hz",
        dest="sim_hz",
        type=float,
        default=None,
        help="Simulation control frequency in Hz (<=0 for unlimited)",
    )
    parser.add_argument(
        "--team-batch",
        action="store_true",
        help="Run one process for all robots and submit batched commands per step.",
    )
    parser.add_argument(
        "--red-team-size",
        dest="red_team_size",
        type=int,
        default=0,
        help="Red team robot count for --team-batch.",
    )
    parser.add_argument(
        "--blue-team-size",
        dest="blue_team_size",
        type=int,
        default=0,
        help="Blue team robot count for --team-batch.",
    )

    # 只解析我们关心的参数；若误带 ROS 参数，不让 argparse 直接报错退出。
    args, unknown = parser.parse_known_args()

    if args.simulation:
        # 仿真模式：不依赖 ROS2。
        import logging

        if args.team_batch:
            runner = TeamSimRunner(args, ip=args.ip, port=args.port, sim_hz=args.sim_hz)
            runner.run()
        else:
            agent = SimAgent(args, ip=args.ip, port=args.port, sim_hz=args.sim_hz)
            agent.run()
    else:
        # 真机模式：必须具备 ROS2 运行环境。
        if not ROS_AVAILABLE:
            print("=" * 60)
            print("[ERROR] ROS2 is not available!")
            print("To run in simulation mode without ROS2, use: --simulation")
            print("To run in ROS2 mode, please install rclpy first.")
            print("=" * 60)
            sys.exit(1)

        rclpy.init(args=sys.argv)
        agent = Agent(sys.argv)

        def signal_handler(sig, frame):
            if agent:
                print("\n")
                agent.get_logger().warning(f"[Core] Interrupted")
                try:
                    agent.stop()
                    agent.destroy_node()
                    rclpy.shutdown()
                except Exception as e:
                    print(
                        f"======== [Core] Error during stop command in signal handler: {e} ========"
                    )
                sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        rclpy.spin(agent)
