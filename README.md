# MotrixArena S2 · 3v3 机器人足球

**中文** | [English](README.en.md)

本队参加 MotrixArena S2 3v3 仿真足球赛的决策代码与 K1 步态模型。策略围绕前锋、支援和守门员协作，处理开球落位、追球推进、边线脱困与门前防守，通过 ZeroMQ 向仿真器发送速度指令。

这是**队伍提交包**，不包含完整物理仿真器、机器人资产或步态训练工程。

## 目录

```text
.
├── decider/
│   ├── decider.py            # 仿真入口；保留原 ROS2 接口
│   ├── user_entry.py         # 本队核心策略
│   ├── config.yaml           # 公共默认配置
│   ├── interfaces/           # 感知、动作、裁判与 ZMQ 通信
│   ├── logic/                # 基础动作及角色状态机
│   ├── strategy/             # 保留的团队管理模块
│   └── requirements.txt
├── gait/                     # K1 AMP ONNX 模型
├── tests/                    # 无需仿真器的回归检查
├── README_决策.md
└── THIRD_PARTY_NOTICES.md
```

## 安装决策端

建议 Python 3.12。以下命令在本仓库根目录执行，仿真模式不需要 ROS2。

```bash
python -m venv .venv
```

Linux / macOS：`source .venv/bin/activate`。Windows PowerShell：`.\.venv\Scripts\Activate.ps1`。

```bash
python -m pip install -r decider/requirements.txt
python decider/decider.py --help
```

## 启动比赛

先在独立环境中按足球仿真框架说明安装并启动仿真器，设置每队 3 台机器人、9 × 6 m 场地及裁判系统。确认该版本支持 K1 AMP `375 → 22` 模型，显式选择 `gait/k1_amp_20260612.onnx` 的绝对路径；安装 Decider 依赖不会启动仿真器。

参考框架的 Sim Manager 入口如下（在**框架根目录**执行）：

```bash
python simulation/motrixsim/sim_manager.py --host 127.0.0.1 --port 8000
```

在 Manager 中获取实际 ZMQ 端口，然后回到本仓库运行。以下 `5555` 仅为示例。

```bash
# 红队 0 号；另开终端将 --id 改为 1、2，组成红队
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0

# 蓝队 0 号；同样可启动 1、2 号
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color blue --id 0
```

若服务端支持 `commands` 批量消息，可用一个进程控制整队；共享的 `TEAM_TRACKERS` 也在此模式下跨机器人生效：

```bash
python decider/decider.py --simulation --team-batch --red-team-size 3 --blue-team-size 0 --ip 127.0.0.1 --port 5555
```

蓝队批量模式使用 `--red-team-size 0 --blue-team-size 3`。不要同时用单机器人与批量进程控制同一台机器人。单机器人多进程模式不共享进程内 tracker。

## 配置与协议

默认 `champion` 为正赛策略模式名称，**不代表获奖或夺冠声明**。其他模式为 `baseline`（对抗基线）、`penalty`（点球联调）。

```bash
# Bash：在启动 Decider 前设置
export MOS_BRAIN_STRATEGY_MODE=champion
```

```powershell
# PowerShell
$env:MOS_BRAIN_STRATEGY_MODE = "champion"
```

本机 IP、队伍编号等写入 `decider/config_override.yaml`，不要修改公开默认配置来保存私人设置。该文件已被 Git 忽略，示例：

```yaml
team_id: 0  # 示例；比赛时填写实际队伍编号
server_ip: "127.0.0.1"
server_port: 5555
```

CLI 的 `--ip` / `--port` 决定仿真连接地址。其他环境变量见[决策说明](README_决策.md)。

```json
{"cmd": [0.5, 0.0, 0.0], "id": 0, "timestamp": 0}
```

`cmd=[vx, vy, w]` 是机器人坐标系归一化速度，范围 `[-1, 1]`。红队全局 ID 为 `0..6`，蓝队为 `7..13`；指定 `--color` 时 `--id` 是队内编号。比赛控制不发送非标准 `kick` 字段。步态由仿真端加载，Decider 不直接运行 ONNX。

## 检查

```bash
python -m compileall -q decider tests
python -m unittest discover -s tests -v
python decider/logic/policy_statemachines/test.py
```

代码格式与基础静态检查：

```bash
python -m pip install ruff==0.17.5
ruff check decider tests
ruff format --check decider tests
```

GitHub Actions 自动运行上述检查及 CLI 帮助检查。这些检查不等于完整 3v3 比赛验证；原交付包引用的 `tools/eval_match.py` 和评测日志未包含，因此不发布未经复现的胜率或比分结论。

## 模型与许可

Decider 沿用 **GPL-3.0-or-later**，保留 MOS-Brain Contributors 版权，见 [LICENSE](LICENSE) 和[第三方声明](THIRD_PARTY_NOTICES.md)。模型权重的来源及再分发许可仍需确认。

运行日志、私人覆盖配置、虚拟环境和 `.local-archive/` 原始备份均已被 `.gitignore` 排除。通过 Git 提交公开文件，不要把整个工作目录连同隐藏备份直接打包上传。
