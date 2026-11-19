# 决策说明

## 入口与职责

- `decider/decider.py`：CLI、单机器人 / 团队批量循环和接口初始化。
- `decider/user_entry.py`：主策略、角色分配、tracker、定位球与运动决策。
- `decider/logic/`：找球、追球、带球、返回场地与守门状态机。
- `decider/interfaces/`：视觉、裁判状态、动作与 ZeroMQ 通信。
- `decider/strategy/team_manager.py`：保留的独立团队管理实现；默认主策略不通过该模块运行。

默认采用 S 场地（9 × 6 m），`champion` 是策略模式名称。策略围绕前锋、支援和守门员协调；`READY` 落位，`SET`、受罚与结束状态停车，`PLAYING` 按球权和区域处理推进与防守。具体行为以代码为准。

## 环境变量

| 变量 | 默认 / 用途 |
| --- | --- |
| `MOS_BRAIN_STRATEGY_MODE` | `champion`；可选 `baseline`、`penalty` |
| `MOS_BRAIN_SUPPORT_CO_PUSH` | 默认关闭；`1` 启用实验性双机夹推 |
| `MOS_BRAIN_DRIBBLE_DEBUG` | 默认关闭；`1` 开启带球调试 |
| `MOS_BRAIN_REPORT_DIR` | `logs/match_reports`，策略报告目录 |
| `MOS_BRAIN_MATCH_CONFIG` | 外部场地配置 JSON 路径 |
| `MOS_BRAIN_SIM_LOG_LEVEL` | `WARNING` |
| `MOS_BRAIN_SIM_FILE_LOG` | 默认关闭；`1` 开启文件日志 |
| `MOS_BRAIN_MUJOCO_COMMON_PATH` | 可选：外部 `zmq_wrapper.py` 所在目录 |

未提供外部通信封装时，使用随包附带的 `interfaces/zmq_wrapper.py`。完整仿真器仍需单独安装。

## 本机配置

`config.yaml` 为共享默认值。个人配置可放在 `config_override.json`、`config_override.yaml` 或 `config_override.yml`，按上述顺序覆盖，后者优先。它们已被 Git 忽略。公开 `team_id` 已替换为示例值 `0`；正式比赛需设置实际队号。

仿真端 IP、端口通过 `--ip`、`--port` 指定；不要假定 YAML 中的地址能覆盖 CLI 默认值。启动及检查命令见[主 README](README.md)。

## 已知边界

- 单机器人进程之间不共享 `TEAM_TRACKERS`；批量模式需要服务端支持 `commands` 消息。
- 比赛输出仅使用 `cmd` 速度控制；带球与射门效果依赖步态和接触模型。
- 原交付包中的本地评测脚本与结果日志缺失，旧实验比分不作为公开效果证明。
- 当前整理仅完成静态检查、回归测试和本地 ZMQ 消息往返；没有运行物理仿真比赛或 ROS2 真机测试。
