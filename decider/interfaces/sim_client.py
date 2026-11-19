# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import time
import json
import logging

# 定位 zmq_wrapper 所在路径。
# 优先级：
# 1. 显式环境变量 MOS_BRAIN_MUJOCO_COMMON_PATH
# 2. 新版 mujoco/common 路径
# 3. 旧版 isaac_sim/common 路径
COMMON_CANDIDATES = []
env_common = os.environ.get("MOS_BRAIN_MUJOCO_COMMON_PATH")
if env_common:
    COMMON_CANDIDATES.append(os.path.abspath(env_common))
COMMON_CANDIDATES.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation/mujoco/src/common"))
)
COMMON_CANDIDATES.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../simulation/isaac_sim/src/common")
    )
)

COMMON_PATH = None
for candidate in COMMON_CANDIDATES:
    if os.path.isdir(candidate):
        COMMON_PATH = candidate
        if COMMON_PATH not in sys.path:
            sys.path.append(COMMON_PATH)
        break

try:
    from zmq_wrapper import ZMQWrapper
    import zmq
except ImportError:
    # Keep the submission usable without a local simulator checkout.
    from interfaces.zmq_wrapper import ZMQWrapper
    import zmq


class SimClient:
    """仿真通信客户端，负责与仿真器做单次请求-响应交互。"""

    def __init__(self, ip="127.0.0.1", port="5555"):
        self.logger = logging.getLogger("SimClient")

        if ZMQWrapper is None:
            raise RuntimeError("ZMQ dependencies missing.")

        self.ip = ip
        self.port = port
        self.addr = f"tcp://{ip}:{port}"

        self.client = ZMQWrapper(socket_type=zmq.REQ, addr=self.addr)
        self.client.connect()
        self.logger.info(f"Connected to Sim Server at {self.addr}")

    def communicate(self, cmd_vel=[0.0, 0.0, 0.0], robot_id=0):
        """
        发送一拍控制命令，并阻塞等待一拍世界状态。

        参数：
        - cmd_vel: [vx, vy, w]
        - robot_id: 仿真器里的全局机器人编号

        返回：
        - 合法的状态字典
        - 或在通信失败时返回 None
        """
        req = {"cmd": cmd_vel, "id": robot_id, "timestamp": time.time()}

        try:
            self.client.send_json(req)
            resp = self.client.recv_json()
            # 基础校验：仿真端至少应返回 state 字段。
            if not resp or "state" not in resp:
                self.logger.warning("Received invalid response from Sim.")
                return None
            return resp
        except Exception as e:
            self.logger.error(f"Communication error: {e}")
            return None

    def communicate_batch(self, commands):
        """
        发送一整队机器人的控制命令，并阻塞等待一拍世界状态。

        参数：
        - commands: [{"id": int, "cmd": [vx, vy, w], "timestamp": float, "source": str}, ...]

        返回：
        - 合法的状态字典
        - 或在通信失败时返回 None
        """
        req = {
            "commands": commands,
            "timestamp": time.time(),
            "source": "team_batch",
        }

        try:
            self.client.send_json(req)
            resp = self.client.recv_json()
            if not resp or "state" not in resp:
                self.logger.warning("Received invalid batch response from Sim.")
                return None
            return resp
        except Exception as e:
            self.logger.error(f"Batch communication error: {e}")
            return None

    def force_reset(self):
        """强制重建 ZMQ socket，主要用于停机和异常恢复。"""
        try:
            self.client.close()
        except:
            pass
        try:
            self.client = ZMQWrapper(socket_type=zmq.REQ, addr=self.addr)
            self.client.connect()
            self.logger.info("Socket force reset complete.")
        except Exception as e:
            self.logger.error(f"Failed to reset socket: {e}")

    def close(self):
        if self.client:
            self.client.close()
