# MotrixArena S2 · 3v3 Robot Soccer

[中文](README.md) | **English**

Our team's decision code and K1 locomotion checkpoint for the MotrixArena S2 3v3 simulated soccer competition. The strategy coordinates striker, support, and goalkeeper roles, with kickoff positioning, ball pursuit, sideline recovery, and goal-area defense. Velocity commands are sent to the simulator over ZeroMQ.

This is a **team submission package**. It does not include the complete physics simulator, robot assets, or locomotion training project.

## Layout

```text
.
├── decider/
│   ├── decider.py            # Simulation entry; original ROS2 interface retained
│   ├── user_entry.py         # Team strategy
│   ├── config.yaml           # Public defaults
│   ├── interfaces/           # Perception, actions, referee, and ZMQ
│   ├── logic/                # Action and role state machines
│   ├── strategy/             # Retained team manager module
│   └── requirements.txt
├── gait/                     # K1 AMP ONNX checkpoint
├── tests/                    # Simulator-free regression checks
├── README_决策.md             # Decision notes (Chinese)
└── THIRD_PARTY_NOTICES.md
```

## Install the Decider

Python 3.12 is recommended. Run these commands from this repository's root. ROS2 is not required for simulation mode.

```bash
python -m venv .venv
```

Linux / macOS: `source .venv/bin/activate`. Windows PowerShell: `.\.venv\Scripts\Activate.ps1`.

```bash
python -m pip install -r decider/requirements.txt
python decider/decider.py --help
```

## Run a match

Install and start the soccer simulator in a separate environment using its documentation. Configure three robots per team, a 9 × 6 m field, and the referee. Verify support for the K1 AMP `375 → 22` interface and explicitly select the absolute path to `gait/k1_amp_20260612.onnx`. Installing the Decider dependencies does not start a simulator.

The reference framework provides this Sim Manager entry point, run from the **framework root**:

```bash
python simulation/motrixsim/sim_manager.py --host 127.0.0.1 --port 8000
```

Obtain the actual ZMQ port from the Manager, then return to this repository. `5555` below is an example.

```bash
# Red player 0; use additional terminals with --id 1 and 2 for the red team
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color red --id 0

# Blue player 0; similarly start players 1 and 2
python decider/decider.py --simulation --ip 127.0.0.1 --port 5555 --color blue --id 0
```

For servers supporting batched `commands` messages, run a whole team in one process. This also allows its robots to share the in-process `TEAM_TRACKERS` state:

```bash
python decider/decider.py --simulation --team-batch --red-team-size 3 --blue-team-size 0 --ip 127.0.0.1 --port 5555
```

For blue, use `--red-team-size 0 --blue-team-size 3`. Do not control the same robot from both single-player and batch processes. Separate single-player processes do not share the in-process tracker.

## Configuration and protocol

`champion` is the default competition strategy's **mode name, not an award claim**. Other modes are `baseline` (comparison strategy) and `penalty` (penalty-shot debugging).

```bash
# Bash, before starting the Decider
export MOS_BRAIN_STRATEGY_MODE=champion
```

```powershell
# PowerShell
$env:MOS_BRAIN_STRATEGY_MODE = "champion"
```

Store private settings in the Git-ignored `decider/config_override.yaml`, for example:

```yaml
team_id: 0  # Example; use the actual competition team ID
server_ip: "127.0.0.1"
server_port: 5555
```

Simulation connection parameters come from CLI `--ip` / `--port`. Additional environment variables include `MOS_BRAIN_REPORT_DIR` (default `logs/match_reports`), `MOS_BRAIN_MATCH_CONFIG` (external match JSON), `MOS_BRAIN_SIM_LOG_LEVEL` (default `WARNING`), and `MOS_BRAIN_SIM_FILE_LOG=1` (file logging). Experimental cooperative pushing is disabled unless `MOS_BRAIN_SUPPORT_CO_PUSH=1` is set.

```json
{"cmd": [0.5, 0.0, 0.0], "id": 0, "timestamp": 0}
```

`cmd=[vx, vy, w]` contains normalized robot-frame velocities in `[-1, 1]`. Global IDs are `0..6` for red and `7..13` for blue; with `--color`, CLI `--id` is a player index within the team. Match control does not send a nonstandard `kick` field. The simulator loads the ONNX policy; the Decider does not run it directly.

## Checks

```bash
python -m compileall -q decider tests
python -m unittest discover -s tests -v
python decider/logic/policy_statemachines/test.py
python -m pip install ruff==0.17.5
ruff check decider tests
ruff format --check decider tests
```

GitHub Actions runs these checks plus CLI help validation. They do not establish full 3v3 match compatibility. The original package referenced `tools/eval_match.py` and match logs that were not supplied, so unverified scores and win rates are not presented as reproducible results.

## Model and license

The supplied model has one float32 input `nn_input [1, 375]` and one float32 output `nn_output [1, 22]`. Export debug strings containing private paths were removed; parameters and computation are unchanged.

Sanitized checkpoint SHA-256:

```text
df26cfb4531bc0dae265a6536e4c1f9e287fe54fd8ab63cc9b492a182afae114
```

The Decider retains **GPL-3.0-or-later** and MOS-Brain Contributors attribution. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). The checkpoint's training provenance and redistribution terms still need confirmation.

Original private backups, the local virtual environment, and caches have been removed from the delivery directory. Logs and private overrides are excluded by `.gitignore`.
