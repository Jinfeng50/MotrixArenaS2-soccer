import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from logic.sub_statemachines import chase_ball, dribble, find_ball, go_back_to_field
from logic.policy_statemachines import goalkeeper


ENABLE_DRIBBLE_DEBUG = os.environ.get("MOS_BRAIN_DRIBBLE_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
STRATEGY_MODE = os.environ.get("MOS_BRAIN_STRATEGY_MODE", "champion").strip().lower()
ENABLE_SUPPORT_CO_PUSH = os.environ.get("MOS_BRAIN_SUPPORT_CO_PUSH", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REPORT_DIR = os.environ.get(
    "MOS_BRAIN_REPORT_DIR",
    os.path.abspath(os.path.join(CUR_DIR, "..", "logs", "match_reports")),
)

TEAM_TRACKERS = {}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle_rad(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def normalize_angle_deg(angle):
    return math.degrees(normalize_angle_rad(math.radians(angle)))


def resolve_field_dimensions(agent):
    cfg = agent.get_config()
    league = str(cfg.get("league", "S"))
    field_size_cfg = cfg.get("field_size", {})
    fallback = field_size_cfg.get(league, field_size_cfg.get("S", [9.0, 6.0]))
    field_length = float(fallback[0])
    field_width = float(fallback[1])

    candidates = []
    env_match_config = os.environ.get("MOS_BRAIN_MATCH_CONFIG")
    if env_match_config:
        candidates.append(os.path.abspath(env_match_config))
    candidates.append(
        os.path.abspath(
            os.path.join(CUR_DIR, "../simulation/motrixsim/assets/config/match_config.json")
        )
    )
    candidates.append(
        os.path.abspath(os.path.join(CUR_DIR, "../simulation/isaac_sim/config/match_config.json"))
    )

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            field = data.get("field", {})
            field_length = float(field.get("length", field_length))
            field_width = float(field.get("width", field_width))
            break
        except Exception:
            continue
    return field_length, field_width


class DataRecorder:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"dribble_debug_{timestamp}.csv")
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.header_written = False

    def log(self, data):
        if not self.header_written:
            self.writer.writerow(data.keys())
            self.header_written = True
        self.writer.writerow(data.values())
        self.file.flush()

    def close(self):
        self.file.close()


class MatchEventRecorder:
    def __init__(self, team_key, strategy_mode):
        os.makedirs(REPORT_DIR, exist_ok=True)
        safe_team_key = team_key.replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(REPORT_DIR, f"{safe_team_key}_{strategy_mode}_{timestamp}.jsonl")
        self.file = open(self.path, "w", encoding="utf-8")

    def log(self, event_type, **payload):
        row = {
            "ts": round(time.time(), 3),
            "event": event_type,
            **payload,
        }
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()


class TeamTracker:
    def __init__(self, team_key, field_length, field_width, strategy_mode):
        self.team_key = team_key
        self.field_length = field_length
        self.field_width = field_width
        self.strategy_mode = strategy_mode
        self.players = {}
        self.team_size = 0
        self.last_ball_map = None
        self.last_ball_update = 0.0
        self.primary_id = None
        self.primary_reason = "unset"
        self.last_gc_state = None
        self.last_score = None
        self.last_opp_score = None
        self.last_set_play = None
        self.last_can_kick = None
        self.last_kicking_team = None
        self.last_clock_log_ts = 0.0
        self.current_set_play = None
        self.current_can_kick = False
        self.current_game_state = None
        self.opponent_kickoff_release_ts = 0.0
        self.own_kickoff_started_ts = 0.0
        self.tactical_memory = {}
        self.recorder = MatchEventRecorder(team_key, strategy_mode)

    def update_player(self, agent):
        local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
        configured_size = int(getattr(agent, "team_size", 0) or 0)
        self.team_size = max(self.team_size, configured_size, local_id + 1)
        self.players[local_id] = {
            "local_id": local_id,
            "role": getattr(agent, "role", "support"),
            "penalty": int(agent.gamecontroller.penalty),
            "game_state": agent.gamecontroller.game_state,
            "pos": tuple(float(v) for v in agent.get_self_pos()),
            "yaw": float(agent.get_self_yaw()),
            "ball_visible": bool(agent.get_if_ball()),
            "ball_dist": float(agent.get_ball_distance()),
            "ball_angle": float(agent.get_ball_angle())
            if agent.get_ball_angle() is not None
            else None,
            "ball_map": tuple(float(v) for v in agent.get_ball_pos_in_map())
            if agent.get_if_ball()
            else None,
            "ts": time.time(),
        }

        if agent.get_if_ball():
            self.last_ball_map = self.players[local_id]["ball_map"]
            self.last_ball_update = self.players[local_id]["ts"]

    def update_match_events(self, agent):
        gc = agent.gamecontroller
        self.current_set_play = gc.set_play_name
        self.current_can_kick = bool(gc.can_kick)
        self.current_game_state = gc.game_state
        if gc.game_state != self.last_gc_state:
            self.recorder.log(
                "game_state",
                state=gc.game_state,
                set_play=gc.set_play_name,
                secs_remaining=gc.secs_remaining,
            )
            self.last_gc_state = gc.game_state

        previous_opponent_kickoff = (
            self.last_set_play == "SET_PLAY_KICK_OFF"
            and self.last_kicking_team is not None
            and self.last_kicking_team != gc.team
        )
        if gc.set_play_name == "SET_PLAY_NONE" and previous_opponent_kickoff:
            self.opponent_kickoff_release_ts = time.time()

        own_kickoff_active = (
            gc.set_play_name == "SET_PLAY_KICK_OFF"
            and bool(gc.can_kick)
            and gc.kicking_team == gc.team
        )
        if own_kickoff_active and self.own_kickoff_started_ts <= 0.0:
            self.own_kickoff_started_ts = time.time()
        elif not own_kickoff_active:
            self.own_kickoff_started_ts = 0.0

        set_play_tuple = (gc.set_play_name, gc.can_kick, gc.kicking_team)
        if set_play_tuple != (self.last_set_play, self.last_can_kick, self.last_kicking_team):
            ball_map = self.get_ball_map()
            payload = {
                "set_play": gc.set_play_name,
                "can_kick": gc.can_kick,
                "kicking_team": gc.kicking_team,
            }
            if ball_map is not None:
                payload["ball_x"] = round(float(ball_map[0]), 3)
                payload["ball_y"] = round(float(ball_map[1]), 3)
            self.recorder.log(
                "set_play",
                **payload,
            )
            self.last_set_play = gc.set_play_name
            self.last_can_kick = gc.can_kick
            self.last_kicking_team = gc.kicking_team

        score_pair = (gc.score, gc.opponent_score)
        if score_pair != (self.last_score, self.last_opp_score):
            self.recorder.log(
                "score",
                score=gc.score,
                opponent_score=gc.opponent_score,
                secs_remaining=gc.secs_remaining,
            )
            self.last_score, self.last_opp_score = score_pair

        now = time.time()
        if now - self.last_clock_log_ts > 0.8:
            self.recorder.log(
                "clock",
                state=gc.game_state,
                set_play=gc.set_play_name,
                secs_remaining=gc.secs_remaining,
                score=gc.score,
                opponent_score=gc.opponent_score,
            )
            self.last_clock_log_ts = now

    def get_ball_map(self):
        if self.last_ball_map is None:
            return None
        if time.time() - self.last_ball_update > 1.0:
            return None
        return self.last_ball_map

    def zone_age(self, key, active):
        now = time.time()
        if not active:
            self.tactical_memory.pop(key, None)
            return 0.0
        started = self.tactical_memory.setdefault(key, now)
        return now - started

    def choose_primary(self):
        candidates = []
        ball_map = self.get_ball_map()
        if ball_map is None:
            self.primary_id = None
            self.primary_reason = "no_ball"
            return None

        for local_id, info in self.players.items():
            if info["penalty"] != 0:
                continue
            px, py = info["pos"]
            dist = math.hypot(ball_map[0] - px, ball_map[1] - py)
            role = info["role"]
            role_bias = {"striker": -0.40, "support": 0.16, "keeper": 1.4}.get(role, 0.0)
            defensive_danger = (
                ball_map[0] < -self.field_length / 2.0 + 1.65 and abs(ball_map[1]) < 1.18
            )
            if defensive_danger:
                role_bias = {"striker": 0.45, "support": 0.32, "keeper": -0.45}.get(role, 0.0)
            own_kickoff_start = (
                self.current_set_play == "SET_PLAY_KICK_OFF"
                and self.current_can_kick
                and abs(ball_map[0]) < 0.55
                and abs(ball_map[1]) < 0.55
            )
            if own_kickoff_start:
                role_bias = {"striker": 0.05, "support": -0.34, "keeper": 1.8}.get(role, 0.0)
            front_bonus = 0.0
            if role == "striker" and ball_map[0] > -0.7 and not own_kickoff_start:
                front_bonus = -0.28
            if role == "support" and ball_map[0] > -0.6 and not own_kickoff_start:
                front_bonus = 0.22
            if ball_map[0] > 1.45 and not defensive_danger:
                behind_ball = px < ball_map[0] - 0.18
                in_front_of_ball = px > ball_map[0] + 0.18
                if role in ("support", "sweeper") and behind_ball and abs(py - ball_map[1]) < 1.25:
                    front_bonus -= 0.44
                if role == "striker" and in_front_of_ball:
                    front_bonus += 0.58
            candidates.append((dist + role_bias + front_bonus, local_id, role, dist))

        if not candidates:
            self.primary_id = None
            self.primary_reason = "all_penalized"
            return None

        candidates.sort(key=lambda item: item[0])

        score, chosen_id, chosen_role, raw_dist = candidates[0]
        candidate_scores = {
            local_id: (score, role, raw_dist) for score, local_id, role, raw_dist in candidates
        }
        self.primary_reason = "closest_bias"
        striker_candidate = None
        for candidate in candidates:
            _, candidate_id, candidate_role, candidate_dist = candidate
            if candidate_role == "striker":
                striker_candidate = candidate
                break
        own_kickoff_start = (
            self.current_set_play == "SET_PLAY_KICK_OFF"
            and self.current_can_kick
            and abs(ball_map[0]) < 0.55
            and abs(ball_map[1]) < 0.55
        )
        defensive_danger = ball_map[0] < -self.field_length / 2.0 + 1.65 and abs(ball_map[1]) < 1.18
        front_takeover_zone = ball_map[0] > 1.45
        if (
            ball_map[0] > -0.7
            and striker_candidate is not None
            and not own_kickoff_start
            and not defensive_danger
            and not front_takeover_zone
        ):
            striker_score, striker_id, striker_role, striker_dist = striker_candidate
            if striker_dist < 1.45 and striker_score <= score + 0.75:
                score, chosen_id, chosen_role, raw_dist = striker_candidate
                self.primary_reason = "attack_striker_priority"

        if self.primary_id in candidate_scores:
            current_score, current_role, current_dist = candidate_scores[self.primary_id]
            if (
                self.primary_reason != "attack_striker_priority"
                and current_role != "keeper"
                and not (current_role == "support" and ball_map[0] > -0.7)
                and not defensive_danger
                and score > current_score - 0.18
            ):
                chosen_id = self.primary_id
                chosen_role = current_role
                raw_dist = current_dist
                self.primary_reason = "hysteresis_hold"
            else:
                if self.primary_reason != "attack_striker_priority":
                    self.primary_reason = "closest_bias"
        else:
            if self.primary_reason != "attack_striker_priority":
                self.primary_reason = "closest_bias"
        if chosen_id != self.primary_id:
            self.recorder.log(
                "primary_switch",
                primary_id=chosen_id,
                role=chosen_role,
                ball_x=round(ball_map[0], 3),
                ball_y=round(ball_map[1], 3),
                ball_dist=round(raw_dist, 3),
                reason=self.primary_reason,
            )
        self.primary_id = chosen_id
        return chosen_id


class AdvancedDribbler:
    def __init__(self, agent):
        self.agent = agent
        self.logger = agent.get_logger().get_child("AdvDribble")
        self.recorder = None
        if ENABLE_DRIBBLE_DEBUG:
            log_dir = os.path.abspath(os.path.join(CUR_DIR, "..", "debug_logs"))
            self.recorder = DataRecorder(log_dir)

        self.bturn_p = 2.0
        self.side_correction_p = 2.8
        self.forward_p = 1.0
        self.setup_dist = 0.34
        self.dribble_dist = 0.16
        self.max_fw_vel = 0.9
        self.field_length, self.field_width = resolve_field_dimensions(agent)
        self.spread_factor_max = 20.0
        self.spread_factor_min = 8.0
        self.turn_to_ball_enter_bx = 0.03
        self.turn_to_ball_exit_bx = 0.08
        self.turn_to_ball_mode = False
        self.direction_consistency_bx = 0.12

    def get_target_vector(self):
        goal_x = self.field_length / 2.0
        goal_y = 0.0
        my_pos = self.agent.get_self_pos()
        if my_pos is None:
            return np.array([1.0, 0.0]), False

        g_dx = goal_x - my_pos[0]
        g_dy = goal_y - my_pos[1]

        dist_to_left = (self.field_width / 2.0) - my_pos[1]
        dist_to_right = my_pos[1] - (-self.field_width / 2.0)
        repulsion_y = 0.0
        margin = 0.7

        if dist_to_left < margin:
            repulsion_y -= (margin - dist_to_left) * 2.2
        if dist_to_right < margin:
            repulsion_y += (margin - dist_to_right) * 2.2

        final_dx = g_dx
        final_dy = g_dy + repulsion_y
        norm = math.hypot(final_dx, final_dy)
        if norm < 1e-6:
            return np.array([1.0, 0.0]), False
        target_vec = np.array([final_dx / norm, final_dy / norm])
        return target_vec, abs(my_pos[1]) < self.field_width / 3.0

    def run(self):
        if not self.agent.get_if_ball():
            self.agent.cmd_vel(0.0, 0.0, 0.0)
            return

        ball_pos = self.agent.get_ball_pos()
        b_x, b_y = float(ball_pos[0]), float(ball_pos[1])
        my_pos = self.agent.get_self_pos()
        my_yaw = self.agent.get_self_yaw()

        ball_dist = math.hypot(b_x, b_y)
        ball_angle_to_robot = math.atan2(b_y, b_x)
        prev_turn_mode = self.turn_to_ball_mode
        if self.turn_to_ball_mode:
            self.turn_to_ball_mode = b_x < self.turn_to_ball_exit_bx
        else:
            self.turn_to_ball_mode = b_x < self.turn_to_ball_enter_bx

        if self.turn_to_ball_mode != prev_turn_mode:
            self.logger.info(
                f"[AdvDribble] mode turn_to_ball={self.turn_to_ball_mode} b=({b_x:.3f},{b_y:.3f})"
            )

        if self.turn_to_ball_mode:
            turn_speed = clamp(1.5 * ball_angle_to_robot, -1.5, 1.5)
            approach_speed = 0.0
            if ball_dist > 0.3:
                approach_speed = clamp(0.3 * b_x / (ball_dist + 0.01), -0.3, 0.5)
            self.agent.cmd_vel(approach_speed, 0.0, turn_speed)
            self.agent.move_head(math.inf, math.inf)
            if self.recorder is not None:
                self.recorder.log(
                    {
                        "time": time.time(),
                        "rx": my_pos[0] if my_pos is not None else 0.0,
                        "ry": my_pos[1] if my_pos is not None else 0.0,
                        "ryaw": my_yaw,
                        "ball_x": b_x,
                        "ball_y": b_y,
                        "cmd_x": approach_speed,
                        "cmd_y": 0.0,
                        "cmd_w": turn_speed,
                        "aligned": 0,
                    }
                )
            return

        target_vec_global, is_safe_zone = self.get_target_vector()
        yaw_rad = math.radians(my_yaw)
        t_local_x = target_vec_global[0] * math.cos(yaw_rad) + target_vec_global[1] * math.sin(
            yaw_rad
        )
        t_local_y = -target_vec_global[0] * math.sin(yaw_rad) + target_vec_global[1] * math.cos(
            yaw_rad
        )
        target_angle_local = math.atan2(t_local_y, t_local_x)
        target_angle_deg = math.degrees(target_angle_local)

        da = -self.bturn_p * target_angle_local
        dist_to_ball = math.hypot(b_x, b_y)
        da *= clamp(dist_to_ball / 0.4, 0.4, 1.0)

        c_r = math.cos(-target_angle_local)
        s_r = math.sin(-target_angle_local)
        b_x_virt = b_x * c_r - b_y * s_r
        b_y_virt = b_x * s_r + b_y * c_r

        err_y = b_y_virt
        cmd_y_virt = self.side_correction_p * err_y
        deadband = self.spread_factor_max if is_safe_zone else self.spread_factor_min
        aligned = abs(target_angle_deg) < deadband and abs(err_y) < 0.1 and b_x_virt > 0.1

        near_goal = my_pos is not None and my_pos[0] > self.field_length * 0.22
        if near_goal:
            aligned = abs(target_angle_deg) < 28.0 and b_x_virt > 0.05

        target_dist = self.dribble_dist if aligned else self.setup_dist
        if not aligned and abs(target_angle_deg) < 25.0:
            ratio = clamp((25.0 - abs(target_angle_deg)) / 20.0, 0.0, 1.0)
            target_dist = self.setup_dist - ratio * (self.setup_dist - self.dribble_dist)

        err_x = b_x_virt - target_dist
        forward_factor = 1.0
        if not aligned:
            angle_err = abs(target_angle_deg)
            forward_factor = 1.0 - min(angle_err / 25.0, 1.0) * 0.6
            if angle_err >= 25.0:
                forward_factor = 0.0
        if near_goal and b_x_virt > 0.05:
            forward_factor = max(forward_factor, 0.85)

        cmd_x_virt = self.forward_p * err_x * forward_factor
        if aligned and cmd_x_virt < 0.55:
            cmd_x_virt = 0.55
        if near_goal and b_x_virt > 0.1 and cmd_x_virt < 0.45:
            cmd_x_virt = 0.45

        cmd_x_virt = clamp(cmd_x_virt, -0.4, self.max_fw_vel)
        cmd_y_virt = clamp(cmd_y_virt, -self.max_fw_vel, self.max_fw_vel)

        c = math.cos(target_angle_local)
        s = math.sin(target_angle_local)
        cmd_x = cmd_x_virt * c - cmd_y_virt * s
        cmd_y = cmd_x_virt * s + cmd_y_virt * c
        lin_norm = math.hypot(cmd_x, cmd_y)
        if lin_norm > self.max_fw_vel:
            scale = self.max_fw_vel / lin_norm
            cmd_x *= scale
            cmd_y *= scale

        da = clamp(da, -1.5, 1.5)
        if (
            b_x < self.direction_consistency_bx
            and abs(ball_angle_to_robot) > 0.2
            and abs(da) > 1e-6
        ):
            da = math.copysign(abs(da), ball_angle_to_robot)

        if self.recorder is not None:
            self.recorder.log(
                {
                    "time": time.time(),
                    "rx": my_pos[0] if my_pos is not None else 0.0,
                    "ry": my_pos[1] if my_pos is not None else 0.0,
                    "ryaw": my_yaw,
                    "ball_x": b_x,
                    "ball_y": b_y,
                    "t_vec_gx": float(target_vec_global[0]),
                    "t_vec_gy": float(target_vec_global[1]),
                    "t_ang_local": target_angle_deg,
                    "cmd_x": cmd_x,
                    "cmd_y": cmd_y,
                    "cmd_w": da,
                    "aligned": int(aligned),
                    "safe_zone": int(is_safe_zone),
                    "err_y": err_y,
                    "err_x": err_x,
                }
            )

        self.agent.cmd_vel(cmd_x, cmd_y, da)
        self.agent.move_head(math.inf, math.inf)


def get_team_key(agent):
    team_id = int(agent.get_config().get("team_id", 0))
    color = str(getattr(agent, "color", agent.get_config().get("color", "red"))).lower()
    return f"{color}:{team_id}"


def get_or_create_tracker(agent):
    team_key = get_team_key(agent)
    tracker = TEAM_TRACKERS.get(team_key)
    if tracker is None:
        field_length, field_width = resolve_field_dimensions(agent)
        tracker = TeamTracker(team_key, field_length, field_width, STRATEGY_MODE)
        TEAM_TRACKERS[team_key] = tracker
    return tracker


def get_role_for_agent(agent):
    local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
    team_size = int(getattr(agent, "team_size", 0) or 0)
    if local_id == 0:
        return "striker"
    if team_size <= 2 and local_id == 1:
        return "sweeper"
    if local_id == 1:
        return "support"
    if local_id == 2:
        return "keeper"
    return "support"


def init(agent):
    agent.get_logger().info("[UserEntry] Initializing match logic")
    agent.chase_ball_machine = chase_ball.ChaseBallStateMachine(agent)
    agent.find_ball_machine = find_ball.FindBallStateMachine(agent)
    agent.go_back_machine = go_back_to_field.GoBackToFieldStateMachine(agent)
    agent.dribble_machine = dribble.DribbleStateMachine(agent)
    agent.goalkeeper_machine = goalkeeper.GoalkeeperStateMachine(agent)
    agent.adv_dribbler = AdvancedDribbler(agent)

    agent.state_machine_runners = {
        "chase_ball": agent.chase_ball_machine.run,
        "find_ball": agent.find_ball_machine.run,
        "go_back_to_field": agent.go_back_machine.run,
        "dribble": agent.dribble_machine.run,
        "adv_dribble": agent.adv_dribbler.run,
        "stop": agent.stop,
        "goalkeeper": agent.goalkeeper_machine.run,
    }
    agent._state_machine_runners = agent.state_machine_runners
    agent.default_chase_distance = (
        agent.get_config().get("chase", {}).get("default_chase_distance", 0.8)
    )
    agent.field_length, agent.field_width = resolve_field_dimensions(agent)
    agent.role = get_role_for_agent(agent)
    agent.team_tracker = get_or_create_tracker(agent)
    agent.last_behavior = None
    agent.last_behavior_ts = 0.0
    agent.last_shot_attempt_ts = 0.0
    agent.last_body_push_attempt_ts = 0.0
    agent.embodied_last_intent = None
    agent.embodied_last_intent_ts = 0.0
    agent.embodied_last_plan = None
    agent.embodied_last_plan_ts = 0.0
    agent.sideline_trap_key = None
    agent.sideline_trap_ts = 0.0
    agent.boundary_recovery_ts = 0.0
    agent.relocate()


def loop(agent):
    try:
        game(agent)
    except Exception as exc:
        agent.get_logger().error(f"[UserEntry] loop error: {exc}")
        traceback.print_exc()


def game(agent):
    tracker = get_or_create_tracker(agent)
    tracker.update_player(agent)
    tracker.update_match_events(agent)
    if STRATEGY_MODE == "baseline":
        _baseline_game(agent, tracker)
    elif STRATEGY_MODE in ("penalty", "qualifier", "qualification"):
        _penalty_game(agent, tracker)
    else:
        _champion_game(agent, tracker)


def _set_behavior(agent, behavior, **payload):
    now = time.time()
    if behavior != agent.last_behavior or now - agent.last_behavior_ts > 1.0:
        pos = agent.get_self_pos()
        if pos is not None and len(pos) >= 2:
            payload.setdefault("robot_x", round(float(pos[0]), 3))
            payload.setdefault("robot_y", round(float(pos[1]), 3))
        payload.setdefault("robot_yaw", round(float(agent.get_self_yaw()), 1))
        if "ball_x" not in payload or "ball_y" not in payload:
            ball_map = _get_team_ball(agent, agent.team_tracker)
            if ball_map is not None:
                payload.setdefault("ball_x", round(float(ball_map[0]), 3))
                payload.setdefault("ball_y", round(float(ball_map[1]), 3))
        agent.team_tracker.recorder.log(
            "behavior",
            robot=int(getattr(agent, "local_id", agent.get_config().get("id", 0))),
            role=agent.role,
            behavior=behavior,
            **payload,
        )
        agent.last_behavior = behavior
        agent.last_behavior_ts = now


def _run_find_or_chase(agent, chase_distance=None):
    if not agent.get_if_ball():
        _set_behavior(agent, "find_ball")
        agent.state_machine_runners["find_ball"]()
        return
    ball_dist = agent.get_ball_distance()
    threshold = agent.default_chase_distance if chase_distance is None else chase_distance
    if ball_dist > threshold:
        _set_behavior(agent, "chase_ball", ball_dist=round(ball_dist, 3))
        agent.state_machine_runners["chase_ball"](distance=threshold)
    else:
        _set_behavior(agent, "adv_dribble", ball_dist=round(ball_dist, 3))
        agent.state_machine_runners["adv_dribble"]()


def _aggressive_pursuit(agent, target_distance=0.5):
    if not agent.get_if_ball():
        _set_behavior(agent, "find_ball")
        agent.state_machine_runners["find_ball"]()
        return False

    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    if ball_dist <= target_distance:
        return False

    ball_angle = math.atan2(b_y, b_x)
    vx = 1.0 + 0.35 * max(0.0, ball_dist - target_distance)
    vy = 1.9 * b_y
    w = 2.25 * ball_angle

    if abs(ball_angle) > 0.85:
        vx *= 0.72
    elif abs(ball_angle) > 0.45:
        vx *= 0.9

    if b_x < 0.05:
        vx = 0.25 if ball_dist > target_distance + 0.25 else 0.0
        vy *= 0.45

    vx = clamp(vx, 0.0, 1.0)
    vy = clamp(vy, -1.0, 1.0)
    w = clamp(w, -1.6, 1.6)
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)
    return True


def _map_vector_to_robot(agent, dx, dy):
    yaw_rad = math.radians(agent.get_self_yaw())
    return (
        dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad),
        -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad),
    )


def _stage_push_to_target(
    agent,
    tracker,
    ball_map,
    field_target_x,
    field_target_y,
    behavior,
    setup_back=0.5,
    ready_dist=0.82,
    min_vx=0.18,
    max_vx=0.95,
    vy_limit=0.82,
):
    if not agent.get_if_ball():
        ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
        target_dx = field_target_x - ball_x
        target_dy = field_target_y - ball_y
        target_norm = max(math.hypot(target_dx, target_dy), 1e-6)
        setup_x = clamp(
            ball_x - setup_back * target_dx / target_norm,
            -agent.field_length / 2.0 + 0.45,
            agent.field_length / 2.0 - 0.25,
        )
        setup_y = clamp(
            ball_y - setup_back * target_dy / target_norm,
            -agent.field_width / 2.0 + 0.18,
            agent.field_width / 2.0 - 0.18,
        )
        _set_behavior(
            agent, f"{behavior}_find_setup", target_x=round(setup_x, 2), target_y=round(setup_y, 2)
        )
        _drive_to_pose(
            agent, setup_x, setup_y, face_x=field_target_x, face_y=field_target_y, stop_dist=0.12
        )
        agent.move_head(math.inf, math.inf)
        return True

    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    pos = agent.get_self_pos()

    push_dx = field_target_x - ball_x
    push_dy = field_target_y - ball_y
    push_norm = max(math.hypot(push_dx, push_dy), 1e-6)
    push_dx /= push_norm
    push_dy /= push_norm

    raw_vx, raw_vy = _map_vector_to_robot(agent, push_dx, push_dy)
    if behavior == "cup_vla_clear_own_wide" and ball_dist < 2.05 and b_x > -0.65:
        side = 1.0 if ball_y > 0.0 else -1.0
        clear_yaw_deg = math.degrees(math.atan2(field_target_y - pos[1], field_target_x - pos[0]))
        clear_err_deg = normalize_angle_deg(clear_yaw_deg - agent.get_self_yaw())
        vx = clamp(0.92 * raw_vx + 0.08 * b_x, 0.12, 0.82)
        vy = clamp(1.18 * raw_vy + 0.22 * b_y - 0.18 * side, -1.0, 1.0)
        w = clamp(
            math.radians(clear_err_deg) * 0.68 + 0.12 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.82,
            0.82,
        )
        _set_behavior(
            agent,
            "cup_vla_clear_own_wide_direct",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True
    rel_x = pos[0] - ball_x
    rel_y = pos[1] - ball_y
    behind = rel_x * push_dx + rel_y * push_dy
    lateral = -rel_x * push_dy + rel_y * push_dx
    ready = (
        ball_dist < ready_dist
        and b_x > -0.18
        and abs(b_y) < 0.62
        and raw_vx > -0.04
        and behind < 0.18
        and abs(lateral) < 0.82
    )
    urgent_own_wide = (
        behavior == "cup_vla_clear_own_wide" and ball_dist < 1.65 and b_x > -0.42 and raw_vx > -0.24
    )
    if urgent_own_wide:
        ready = True
    if not ready:
        setup_x = clamp(ball_x - setup_back * push_dx, -half_len + 0.45, half_len - 0.22)
        setup_y = clamp(ball_y - setup_back * push_dy, -half_wid + 0.16, half_wid - 0.16)
        _set_behavior(
            agent,
            f"{behavior}_setup",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            behind=round(behind, 2),
            lateral=round(lateral, 2),
        )
        _drive_to_pose(
            agent, setup_x, setup_y, face_x=field_target_x, face_y=field_target_y, stop_dist=0.1
        )
        agent.move_head(math.inf, math.inf)
        return True

    push_yaw_deg = math.degrees(math.atan2(field_target_y - pos[1], field_target_x - pos[0]))
    push_err_deg = normalize_angle_deg(push_yaw_deg - agent.get_self_yaw())
    vx = clamp(1.12 * raw_vx + 0.06 * b_x, min_vx, max_vx)
    vy = clamp(0.92 * raw_vy + 0.10 * b_y, -vy_limit, vy_limit)
    w = clamp(
        math.radians(push_err_deg) * 0.48 + 0.08 * math.atan2(b_y, max(b_x, 0.04)), -0.62, 0.62
    )
    _set_behavior(
        agent,
        behavior,
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        raw_vx=round(raw_vx, 2),
        raw_vy=round(raw_vy, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)
    return True


def _stage_nonkeeper_handler_role(tracker, ball_map):
    half_len = tracker.field_length / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    if _opponent_kickoff_release_window(tracker, ball_map):
        return "support"
    distances = {}
    for info in tracker.players.values():
        role = info["role"]
        if role == "keeper" or info["penalty"] != 0:
            continue
        px, py = info["pos"]
        distances[role] = min(distances.get(role, 99.0), math.hypot(ball_x - px, ball_y - py))

    striker_dist = distances.get("striker", 99.0)
    support_dist = distances.get("support", distances.get("sweeper", 99.0))

    if ball_x < -half_len + 1.35 and abs(ball_y) > 1.35:
        return "support" if support_dist <= striker_dist + 0.35 else "striker"
    if ball_x < -0.45:
        return "striker" if striker_dist + 0.65 < support_dist else "support"
    if ball_x > 2.35:
        return "support" if support_dist + 0.45 < striker_dist else "striker"
    return "support" if support_dist + 0.25 < striker_dist else "striker"


def _opponent_kickoff_release_window(tracker, ball_map):
    if time.time() - getattr(tracker, "opponent_kickoff_release_ts", 0.0) > 12.0:
        return False
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    return -3.35 < ball_x < -0.75 and abs(ball_y) < 1.25


def _stage_offball_lane(agent, tracker, ball_map):
    half_len = agent.field_length / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])

    if _opponent_kickoff_release_window(tracker, ball_map) and agent.role == "striker":
        target_x = clamp(ball_x - 0.45, -3.2, -1.25)
        target_y = clamp(ball_y * 0.45, -0.75, 0.75)
        _set_behavior(
            agent,
            "stage_opp_kickoff_center_screen",
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.13)
        return

    if ball_x < -half_len + 1.35 and abs(ball_y) > 1.35:
        side = 1.0 if ball_y > 0.0 else -1.0
        if agent.role == "striker":
            target_x = -1.05
            target_y = clamp(-0.35 * side, -0.55, 0.55)
            behavior = "stage_def_outlet_lane"
        else:
            target_x = -2.25
            target_y = clamp(0.45 * side, -0.75, 0.75)
            behavior = "stage_def_center_screen"
        _set_behavior(agent, behavior, target_x=round(target_x, 2), target_y=round(target_y, 2))
        _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.14)
        return

    if agent.role == "striker":
        if ball_x < -0.45:
            target_x = clamp(ball_x + 1.55, -0.1, 1.35)
            target_y = clamp(-0.72 if ball_y >= 0.0 else 0.72, -0.95, 0.95)
            behavior = "stage_striker_counter_lane"
        elif ball_x < 1.6:
            target_x = clamp(ball_x + 1.45, 1.0, 2.65)
            target_y = clamp(ball_y * 0.22, -0.55, 0.55)
            behavior = "stage_striker_receive_lane"
        else:
            target_x = clamp(ball_x + 0.75, 2.9, half_len - 0.38)
            target_y = clamp(ball_y * 0.18, -0.42, 0.42)
            behavior = "stage_striker_finish_lane"
    else:
        if ball_x < -0.45:
            target_x = clamp(ball_x + 0.55, -half_len + 0.85, -0.45)
            target_y = clamp(ball_y * 0.72, -1.25, 1.25)
            behavior = "stage_support_defense_screen"
        elif ball_x < 1.6:
            target_x = clamp(ball_x - 0.55, -0.65, 1.25)
            target_y = clamp(ball_y * 0.35, -0.8, 0.8)
            behavior = "stage_support_second_ball"
        else:
            side = 1.0 if ball_y > 0.0 else -1.0
            target_x = clamp(ball_x - 0.35, 2.15, half_len - 0.45)
            target_y = clamp(side * 0.38 if abs(ball_y) > 0.75 else ball_y * 0.15, -0.55, 0.55)
            behavior = "stage_support_goalmouth_lane"

    _set_behavior(agent, behavior, target_x=round(target_x, 2), target_y=round(target_y, 2))
    _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.14)


def _stage_ball_handler(agent, tracker, ball_map):
    half_len = agent.field_length / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    def_wide_corner = ball_x < -half_len + 1.35 and abs(ball_y) > 1.35

    if agent.role == "keeper":
        if def_wide_corner:
            side = 1.0 if ball_y > 0.0 else -1.0
            target_x = -half_len + 0.34
            target_y = clamp(0.72 * side, -0.85, 0.85)
            _set_behavior(
                agent,
                "stage_keeper_wide_guard",
                ball_x=round(ball_x, 2),
                ball_y=round(ball_y, 2),
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
            )
            _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.1)
            return
        if agent.get_if_ball() and ball_x < -half_len + 1.85 and agent.get_ball_distance() < 1.55:
            _set_behavior(
                agent, "stage_keeper_clearance", ball_dist=round(agent.get_ball_distance(), 3)
            )
            _keeper_clear_ball(agent, tracker, ball_map)
            return
        _keeper_anchor(agent, tracker)
        return

    if _own_kickoff_opening(agent, tracker, ball_map):
        return

    if _finish_attack(agent, tracker, ball_map):
        _set_behavior(
            agent,
            "stage_finish_attack",
            ball_dist=round(agent.get_ball_distance(), 3) if agent.get_if_ball() else None,
        )
        return

    if _opponent_kickoff_release_window(tracker, ball_map):
        target_x = 0.95
        target_y = clamp(ball_y * 0.14, -0.42, 0.42)
        _stage_push_to_target(
            agent,
            tracker,
            ball_map,
            target_x,
            target_y,
            "stage_opp_kickoff_clear",
            setup_back=0.42,
            ready_dist=0.88,
            min_vx=0.34,
            max_vx=1.0,
            vy_limit=0.72,
        )
        return

    if def_wide_corner:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = -0.85
        target_y = clamp(-0.22 * side, -0.38, 0.38)
        _stage_push_to_target(
            agent,
            tracker,
            ball_map,
            target_x,
            target_y,
            "stage_def_wide_clear",
            setup_back=0.46,
            ready_dist=0.82,
            min_vx=0.36,
            max_vx=1.0,
            vy_limit=0.84,
        )
        return

    if ball_x < -half_len + 1.35:
        target_x = -1.25
        target_y = clamp(ball_y * 0.16, -0.35, 0.35)
        _stage_push_to_target(
            agent,
            tracker,
            ball_map,
            target_x,
            target_y,
            "stage_box_clear",
            setup_back=0.44,
            ready_dist=0.72,
            min_vx=0.34,
            max_vx=0.95,
            vy_limit=0.62,
        )
        return

    if ball_x < -0.45:
        target_x = 0.85
        target_y = clamp(ball_y * 0.22, -0.55, 0.55)
        _stage_push_to_target(
            agent,
            tracker,
            ball_map,
            target_x,
            target_y,
            "stage_defensive_progress",
            setup_back=0.48,
            ready_dist=0.78,
            min_vx=0.28,
            max_vx=0.9,
            vy_limit=0.7,
        )
        return

    if ball_x < 2.0:
        target_x = 3.25
        target_y = clamp(ball_y * 0.16, -0.42, 0.42)
        _stage_push_to_target(
            agent,
            tracker,
            ball_map,
            target_x,
            target_y,
            "stage_midfield_drive",
            setup_back=0.5,
            ready_dist=0.84,
            min_vx=0.28,
            max_vx=0.96,
            vy_limit=0.72,
        )
        return

    if abs(ball_y) > 1.05:
        target_x = half_len - (0.18 if ball_x > half_len - 0.95 else 0.35)
        target_y = clamp((1.0 if ball_y > 0.0 else -1.0) * 0.46, -0.56, 0.56)
        _stage_push_to_target(
            agent,
            tracker,
            ball_map,
            target_x,
            target_y,
            "stage_wide_cross",
            setup_back=0.42,
            ready_dist=0.9,
            min_vx=0.08,
            max_vx=0.55,
            vy_limit=1.0,
        )
        return

    target_x = half_len + 0.75
    target_y = clamp(ball_y * 0.04, -0.12, 0.12)
    _stage_push_to_target(
        agent,
        tracker,
        ball_map,
        target_x,
        target_y,
        "stage_goalmouth_drive",
        setup_back=0.38,
        ready_dist=0.95,
        min_vx=0.42,
        max_vx=1.0,
        vy_limit=0.48,
    )


def _stage_match_game(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        if agent.role == "keeper":
            _keeper_anchor(agent, tracker)
        else:
            _set_behavior(agent, "stage_find_ball")
            agent.state_machine_runners["find_ball"]()
        return

    if agent.role == "keeper":
        _stage_ball_handler(agent, tracker, ball_map)
        return

    handler_role = _stage_nonkeeper_handler_role(tracker, ball_map)
    is_handler = agent.role == handler_role or (
        handler_role == "support" and agent.role == "sweeper"
    )
    if is_handler:
        _stage_ball_handler(agent, tracker, ball_map)
        return

    _stage_offball_lane(agent, tracker, ball_map)


def _legacy_attack_open_play(agent, tracker):
    local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
    if tracker.primary_id == local_id:
        _primary_ball_handler(agent, tracker)
        return

    if agent.role == "keeper":
        _keeper_anchor(agent, tracker)
    elif agent.role == "sweeper":
        _sweeper_anchor(agent, tracker)
    else:
        _support_anchor(agent, tracker)


def _finish_attack(agent, tracker, ball_map):
    if not agent.get_if_ball():
        return False
    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    goal_x = agent.field_length / 2.0
    own_goal_x = -agent.field_length / 2.0
    near_goal = ball_map[0] > goal_x - 2.8
    danger = ball_map[0] < own_goal_x + 2.0
    progress_zone = ball_map[0] > own_goal_x + 2.2

    finish_reach = 1.18 if ball_map[0] > goal_x - 0.9 else 1.05
    if ball_dist > finish_reach:
        return False

    if abs(ball_map[0]) < 0.45 and abs(ball_map[1]) < 0.45 and b_x > -0.12 and ball_dist < 0.86:
        yaw_err_deg = normalize_angle_deg(0.0 - agent.get_self_yaw())
        if abs(b_y) > 0.24 or b_x < 0.12:
            vx = 0.22 if b_x > 0.0 else 0.08
            vy = clamp(0.62 * b_y - 0.18 * ball_map[1], -0.34, 0.34)
            w = clamp(math.radians(yaw_err_deg) * 0.34, -0.28, 0.28)
        else:
            vx = 1.0
            vy = clamp(0.08 * b_y - 0.12 * ball_map[1], -0.12, 0.12)
            w = clamp(
                math.radians(yaw_err_deg) * 0.30 + 0.06 * math.atan2(b_y, max(b_x, 0.04)),
                -0.28,
                0.28,
            )
        _set_behavior(
            agent,
            "body_push_kickoff_drive",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            yaw_err=round(yaw_err_deg, 1),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if (
        -0.35 < ball_map[0] < 0.82
        and 0.24 < abs(ball_map[1]) < 1.32
        and ball_dist < 1.05
        and b_x > -0.24
    ):
        side = 1.0 if ball_map[1] > 0.0 else -1.0
        target_x = 2.15
        target_y = side * 0.42
        recover_dx = target_x - ball_map[0]
        recover_dy = target_y - ball_map[1]
        recover_norm = max(math.hypot(recover_dx, recover_dy), 1e-6)
        recover_dx /= recover_norm
        recover_dy /= recover_norm
        pos = agent.get_self_pos()
        recover_setup_x = clamp(ball_map[0] - 0.48 * recover_dx, -0.35, 0.48)
        recover_setup_y = clamp(ball_map[1] - 0.48 * recover_dy, -1.25, 1.25)
        recover_ready = b_x > 0.06 and abs(b_y) < 0.42 and ball_dist < 0.82
        if not recover_ready:
            _set_behavior(
                agent,
                "body_push_center_recover_setup",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                target_x=round(recover_setup_x, 2),
                target_y=round(recover_setup_y, 2),
            )
            _drive_to_pose(
                agent,
                recover_setup_x,
                recover_setup_y,
                face_x=target_x,
                face_y=target_y,
                stop_dist=0.1,
            )
            agent.move_head(math.inf, math.inf)
            return True

        raw_vx, raw_vy = _map_vector_to_robot(agent, recover_dx, recover_dy)
        recover_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        recover_err_deg = normalize_angle_deg(recover_yaw_deg - agent.get_self_yaw())
        vx = clamp(1.2 * raw_vx, 0.58, 1.0)
        vy = clamp(0.42 * raw_vy + 0.04 * b_y, -0.28, 0.28)
        w = clamp(
            math.radians(recover_err_deg) * 0.42 + 0.07 * math.atan2(b_y, max(b_x, 0.04)),
            -0.48,
            0.48,
        )
        _set_behavior(
            agent,
            "body_push_center_recover_forward",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    return _body_push_to_goal(agent, tracker, ball_map, b_x, b_y, ball_dist)


def _body_push_to_goal(agent, tracker, ball_map, b_x, b_y, ball_dist):
    goal_x = agent.field_length / 2.0
    goal_y = 0.0
    half_len = agent.field_length / 2.0
    own_goal_x = -half_len
    half_wid = agent.field_width / 2.0
    pos = agent.get_self_pos()
    yaw_deg = agent.get_self_yaw()
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    to_goal_x = goal_x - ball_x
    to_goal_y = goal_y - ball_y
    to_goal_norm = max(math.hypot(to_goal_x, to_goal_y), 1e-6)
    dir_x = to_goal_x / to_goal_norm
    dir_y = to_goal_y / to_goal_norm
    goal_yaw_deg = math.degrees(math.atan2(goal_y - pos[1], goal_x - pos[0]))
    yaw_err_deg = normalize_angle_deg(goal_yaw_deg - yaw_deg)
    yaw_rad = math.radians(yaw_deg)
    robot_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))

    if time.time() - agent.last_body_push_attempt_ts > 1.0:
        tracker.recorder.log(
            "body_push_attempt",
            robot=robot_id,
            role=agent.role,
            ball_x=round(ball_map[0], 3),
            ball_y=round(ball_map[1], 3),
            ball_dist=round(ball_dist, 3),
            yaw_err=round(yaw_err_deg, 1),
        )
        agent.last_body_push_attempt_ts = time.time()

    behind_x = clamp(ball_x - 0.42 * dir_x, -half_len + 0.45, goal_x - 0.35)
    behind_y = clamp(ball_y - 0.42 * dir_y, -half_wid + 0.65, half_wid - 0.65)
    rel_robot_x = pos[0] - ball_x
    rel_robot_y = pos[1] - ball_y
    behind_error = rel_robot_x * dir_x + rel_robot_y * dir_y
    lateral_error = -rel_robot_x * dir_y + rel_robot_y * dir_x
    behind_dist = math.hypot(pos[0] - behind_x, pos[1] - behind_y)

    goal_inner_y = 0.82
    goal_edge_y = 0.95
    robot_out_front = pos[0] > half_len - 0.04
    robot_out_side = abs(pos[1]) > half_wid - 0.04
    if (
        robot_out_front
        and ball_x > goal_x - 0.18
        and abs(ball_y) > half_wid - 0.42
        and ball_dist < 0.88
    ):
        side = 1.0 if ball_y > 0.0 else -1.0
        vx = -0.02 if b_x < 0.02 else 0.04
        vy = clamp(-1.15 * side + 0.18 * b_y, -1.0, 1.0)
        face_yaw_deg = math.degrees(math.atan2(-pos[1], max(goal_x - 0.35 - pos[0], 0.08)))
        yaw_err = normalize_angle_deg(face_yaw_deg - yaw_deg)
        w = clamp(
            math.radians(yaw_err) * 0.68 + 0.10 * math.atan2(b_y, max(abs(b_x), 0.08)), -0.72, 0.72
        )
        _set_behavior(
            agent,
            "body_push_front_corner_sweep_in",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True
    if robot_out_front and ball_x > goal_x - 0.55:
        side_pull = -0.45 * (1.0 if pos[1] > 0.0 else -1.0)
        vx = -0.10
        vy = clamp(-0.62 * pos[1] + side_pull, -0.72, 0.72)
        face_yaw_deg = math.degrees(math.atan2(-pos[1], max(goal_x - 0.55 - pos[0], 0.08)))
        yaw_err = normalize_angle_deg(face_yaw_deg - yaw_deg)
        w = clamp(math.radians(yaw_err) * 0.72, -0.62, 0.62)
        _set_behavior(
            agent,
            "body_push_front_boundary_hold",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True
    if robot_out_side and ball_x > 1.0:
        side = 1.0 if pos[1] > 0.0 else -1.0
        vx = 0.08 if pos[0] < half_len - 0.5 else -0.04
        vy = clamp(-0.95 * side, -0.95, 0.95)
        face_yaw_deg = math.degrees(math.atan2(-pos[1], max(goal_x - pos[0], 0.25)))
        yaw_err = normalize_angle_deg(face_yaw_deg - yaw_deg)
        w = clamp(math.radians(yaw_err) * 0.65, -0.62, 0.62)
        _set_behavior(
            agent,
            "body_push_side_boundary_hold",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 1.05 and abs(ball_y) < 1.85 and ball_dist < 1.26 and b_x > -0.30:
        side = 1.0 if ball_y > 0.0 else -1.0
        slot_y = side * 0.62 if abs(ball_y) > 0.78 else 0.0
        slot_yaw_deg = math.degrees(math.atan2(slot_y - pos[1], goal_x + 0.85 - pos[0]))
        slot_err_deg = normalize_angle_deg(slot_yaw_deg - yaw_deg)
        vx = 1.0 if b_x > -0.06 else 0.66
        vy = clamp(0.22 * b_y - 0.32 * (ball_y - slot_y), -0.46, 0.46)
        if abs(ball_y) < 0.55:
            vy = clamp(0.18 * b_y - 0.18 * ball_y, -0.22, 0.22)
        w = clamp(
            math.radians(slot_err_deg) * 0.36 + 0.10 * math.atan2(b_y, max(b_x, 0.04)), -0.36, 0.36
        )
        _set_behavior(
            agent,
            "body_push_goalmouth_slot_crash",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_y=round(slot_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 0.42 and abs(ball_y) < 1.15 and ball_dist < 1.22 and b_x > -0.32:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x + 0.48
        target_y = side * 0.18
        punch_dx = target_x - ball_x
        punch_dy = target_y - ball_y
        punch_norm = max(math.hypot(punch_dx, punch_dy), 1e-6)
        punch_dx /= punch_norm
        punch_dy /= punch_norm
        raw_vx, raw_vy = _map_vector_to_robot(agent, punch_dx, punch_dy)
        punch_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        punch_err_deg = normalize_angle_deg(punch_yaw_deg - yaw_deg)
        vx = clamp(0.98 * raw_vx + 0.05 * b_x, 0.22, 0.78)
        vy = clamp(0.92 * raw_vy + 0.12 * b_y, -0.72, 0.72)
        w = clamp(
            math.radians(punch_err_deg) * 0.5 + 0.08 * math.atan2(b_y, max(b_x, 0.04)), -0.58, 0.58
        )
        _set_behavior(
            agent,
            "body_push_goal_line_punch_in",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 1.05 and 1.18 < abs(ball_y) < 2.55 and ball_dist < 1.34 and b_x > -0.46:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x + 0.42
        target_y = side * (0.12 if abs(ball_y) < 1.75 else 0.22)
        cut_dx = target_x - ball_x
        cut_dy = target_y - ball_y
        cut_norm = max(math.hypot(cut_dx, cut_dy), 1e-6)
        cut_dx /= cut_norm
        cut_dy /= cut_norm
        raw_vx, raw_vy = _map_vector_to_robot(agent, cut_dx, cut_dy)
        cut_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        cut_err_deg = normalize_angle_deg(cut_yaw_deg - yaw_deg)
        vx = clamp(0.98 * raw_vx + 0.05 * b_x, 0.16, 0.86)
        vy = clamp(1.45 * raw_vy + 0.14 * b_y - 0.10 * side, -1.0, 1.0)
        w = clamp(
            math.radians(cut_err_deg) * 0.82 + 0.10 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.86,
            0.86,
        )
        _set_behavior(
            agent,
            "body_push_wide_goalmouth_cutback_finish",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if (
        ball_x > goal_x - 0.85
        and goal_edge_y < abs(ball_y) < 2.35
        and ball_dist < 1.35
        and b_x > -0.42
    ):
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x + 0.36
        target_y = side * 0.18
        if ball_x > goal_x - 0.2:
            target_y = side * 0.10
        if abs(ball_y) > 1.55:
            target_x = goal_x + 0.12
            target_y = side * 0.24
        sweep_dx = target_x - ball_x
        sweep_dy = target_y - ball_y
        sweep_norm = max(math.hypot(sweep_dx, sweep_dy), 1e-6)
        sweep_dx /= sweep_norm
        sweep_dy /= sweep_norm
        sweep_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        sweep_err_deg = normalize_angle_deg(sweep_yaw_deg - yaw_deg)
        raw_vx, raw_vy = _map_vector_to_robot(agent, sweep_dx, sweep_dy)
        vx = clamp(1.06 * raw_vx + 0.04 * b_x, 0.08, 0.66)
        if ball_x > goal_x - 0.08:
            vx = clamp(vx, 0.06, 0.42)
        vy = clamp(1.42 * raw_vy + 0.12 * b_y - 0.08 * side, -1.0, 1.0)
        w = clamp(
            math.radians(sweep_err_deg) * 0.72 + 0.09 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.78,
            0.78,
        )
        _set_behavior(
            agent,
            "body_push_near_post_punch_in",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 1.05 and 0.72 < abs(ball_y) < half_wid - 0.18 and ball_dist < 1.42:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x - 0.08
        target_y = side * 0.64
        if abs(ball_y) < 1.25:
            target_x = goal_x + 0.10
            target_y = side * 0.22
        sweep_dx = target_x - ball_x
        sweep_dy = target_y - ball_y
        sweep_norm = max(math.hypot(sweep_dx, sweep_dy), 1e-6)
        sweep_dx /= sweep_norm
        sweep_dy /= sweep_norm
        rel_x = pos[0] - ball_x
        rel_y = pos[1] - ball_y
        sweep_behind = rel_x * sweep_dx + rel_y * sweep_dy
        sweep_lateral = -rel_x * sweep_dy + rel_y * sweep_dx
        raw_vx = sweep_dx * math.cos(yaw_rad) + sweep_dy * math.sin(yaw_rad)
        raw_vy = -sweep_dx * math.sin(yaw_rad) + sweep_dy * math.cos(yaw_rad)
        sweep_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        sweep_err_deg = normalize_angle_deg(sweep_yaw_deg - yaw_deg)
        if sweep_behind > -0.12 or abs(sweep_lateral) > 0.42 or b_x < -0.08:
            setup_x = clamp(ball_x - 0.38 * sweep_dx, goal_x - 1.25, goal_x - 0.22)
            setup_y = clamp(ball_y - 0.38 * sweep_dy, -half_wid + 0.28, half_wid - 0.28)
            _set_behavior(
                agent,
                "body_push_deep_wide_setup",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                behind=round(sweep_behind, 2),
                lateral=round(sweep_lateral, 2),
            )
            _drive_to_pose(
                agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.09
            )
            agent.move_head(math.inf, math.inf)
            return True
        if raw_vx < 0.10 or b_x < -0.18:
            vx = clamp(0.22 * raw_vx, -0.08, 0.16)
            vy = clamp(0.92 * raw_vy + 0.12 * b_y, -0.72, 0.72)
            w = clamp(math.radians(sweep_err_deg) * 1.05, -0.95, 0.95)
            behavior = "body_push_deep_wide_turn_cross"
        else:
            vx_limit = 0.62 if ball_x > goal_x - 0.22 else 0.9
            vx = clamp(1.18 * raw_vx, 0.28, vx_limit)
            vy = clamp(0.92 * raw_vy + 0.10 * b_y, -0.72, 0.72)
            w = clamp(
                math.radians(sweep_err_deg) * 0.62 + 0.14 * math.atan2(b_y, max(b_x, 0.04)),
                -0.72,
                0.72,
            )
            behavior = "body_push_deep_wide_cross"
        _set_behavior(
            agent,
            behavior,
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if 2.0 < ball_x < goal_x - 0.75 and 1.52 < abs(ball_y) < half_wid - 0.35 and ball_dist < 1.12:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x + 0.05
        target_y = side * (0.58 if ball_x > goal_x - 1.7 else 0.72)
        press_dx = target_x - ball_x
        press_dy = target_y - ball_y
        press_norm = max(math.hypot(press_dx, press_dy), 1e-6)
        press_dx /= press_norm
        press_dy /= press_norm
        rel_x = pos[0] - ball_x
        rel_y = pos[1] - ball_y
        press_behind = rel_x * press_dx + rel_y * press_dy
        press_lateral = -rel_x * press_dy + rel_y * press_dx
        raw_vx = press_dx * math.cos(yaw_rad) + press_dy * math.sin(yaw_rad)
        raw_vy = -press_dx * math.sin(yaw_rad) + press_dy * math.cos(yaw_rad)
        press_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        press_err_deg = normalize_angle_deg(press_yaw_deg - yaw_deg)
        if ball_dist < 0.82 and b_x > -0.34:
            vx = clamp(1.18 * raw_vx, 0.34, 1.0)
            vy = clamp(1.45 * raw_vy - 0.16 * side + 0.14 * b_y, -1.0, 1.0)
            w = clamp(
                math.radians(press_err_deg) * 0.95 + 0.12 * math.atan2(b_y, max(b_x, 0.04)),
                -0.92,
                0.92,
            )
            _set_behavior(
                agent,
                "body_push_high_wide_cut_in",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return True

        if press_behind > -0.12 or abs(press_lateral) > 0.50 or raw_vx < 0.02:
            if ball_x > 2.05 and abs(ball_y) > 1.95 and ball_dist < 1.08 and b_x > -0.42:
                forward_bias = 0.68 if ball_x > 2.75 else 0.54
                vx = forward_bias if b_x > -0.05 else 0.38
                vy = clamp(0.92 * side + 0.14 * b_y, -1.0, 1.0)
                w = clamp(0.30 * side + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.58, 0.58)
                _set_behavior(
                    agent,
                    "body_push_high_wide_force_in",
                    ball_dist=round(ball_dist, 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    local_x=round(b_x, 2),
                    local_y=round(b_y, 2),
                    cmd_x=round(vx, 2),
                    cmd_y=round(vy, 2),
                )
                agent.cmd_vel(vx, vy, w)
                agent.move_head(math.inf, math.inf)
                return True
            if raw_vx < 0.08 and ball_dist < 0.72:
                vx = clamp(0.20 * raw_vx, -0.05, 0.16)
                vy = clamp(1.1 * raw_vy - 0.10 * side + 0.10 * b_y, -1.0, 1.0)
                w = clamp(math.radians(press_err_deg) * 1.15, -1.0, 1.0)
                _set_behavior(
                    agent,
                    "body_push_high_wide_turn_in",
                    ball_dist=round(ball_dist, 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    local_x=round(b_x, 2),
                    local_y=round(b_y, 2),
                    raw_vx=round(raw_vx, 2),
                    cmd_x=round(vx, 2),
                    cmd_y=round(vy, 2),
                )
                agent.cmd_vel(vx, vy, w)
                agent.move_head(math.inf, math.inf)
                return True
            setup_x = clamp(ball_x - 0.28 * press_dx, 1.55, goal_x - 0.62)
            setup_y = clamp(ball_y - 0.68 * side, -half_wid + 1.0, half_wid - 1.0)
            _set_behavior(
                agent,
                "body_push_high_wide_setup",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                behind=round(press_behind, 2),
                lateral=round(press_lateral, 2),
            )
            _drive_to_pose(
                agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.09
            )
            agent.move_head(math.inf, math.inf)
            return True
        vx = clamp(1.2 * raw_vx, 0.55, 1.0)
        vy = clamp(1.25 * raw_vy - 0.08 * side + 0.10 * b_y, -1.0, 1.0)
        w = clamp(
            math.radians(press_err_deg) * 0.78 + 0.10 * math.atan2(b_y, max(b_x, 0.04)), -0.82, 0.82
        )
        _set_behavior(
            agent,
            "body_push_high_wide_diagonal",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x < own_goal_x + 1.15 and abs(ball_y) > half_wid - 0.65 and ball_dist < 1.18:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = own_goal_x + 2.55
        target_y = side * 0.22
        clear_dx = target_x - ball_x
        clear_dy = target_y - ball_y
        clear_norm = max(math.hypot(clear_dx, clear_dy), 1e-6)
        clear_dx /= clear_norm
        clear_dy /= clear_norm
        raw_vx = clear_dx * math.cos(yaw_rad) + clear_dy * math.sin(yaw_rad)
        raw_vy = -clear_dx * math.sin(yaw_rad) + clear_dy * math.cos(yaw_rad)
        clear_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        clear_err_deg = normalize_angle_deg(clear_yaw_deg - yaw_deg)
        vx = clamp(1.08 * raw_vx, -0.08, 0.85)
        vy = clamp(1.25 * raw_vy + 0.18 * b_y, -1.0, 1.0)
        if raw_vx > 0.18 and b_x > -0.16:
            vx = max(vx, 0.42)
        w = clamp(
            math.radians(clear_err_deg) * 0.72 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.8, 0.8
        )
        _set_behavior(
            agent,
            "body_push_own_endline_corner_escape",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x < -2.15 and abs(ball_y) > 2.05 and ball_dist < 1.15:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = clamp(ball_x + 2.35, -1.65, 0.35)
        target_y = side * 0.36
        flank_dx = target_x - ball_x
        flank_dy = target_y - ball_y
        flank_norm = max(math.hypot(flank_dx, flank_dy), 1e-6)
        flank_dx /= flank_norm
        flank_dy /= flank_norm
        rel_x = pos[0] - ball_x
        rel_y = pos[1] - ball_y
        flank_behind = rel_x * flank_dx + rel_y * flank_dy
        flank_lateral = -rel_x * flank_dy + rel_y * flank_dx
        raw_vx = flank_dx * math.cos(yaw_rad) + flank_dy * math.sin(yaw_rad)
        raw_vy = -flank_dx * math.sin(yaw_rad) + flank_dy * math.cos(yaw_rad)
        flank_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        flank_err_deg = normalize_angle_deg(flank_yaw_deg - yaw_deg)
        if ball_dist < 0.7 and b_x > -0.18:
            vx = 0.62 if b_x > 0.04 else 0.32
            vy = clamp(-1.15 * side + 0.18 * b_y, -1.0, 1.0)
            w = clamp(-0.38 * side + 0.18 * math.atan2(b_y, max(b_x, 0.04)), -0.72, 0.72)
            _set_behavior(
                agent,
                "body_push_own_flank_inside_escape",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return True

        if flank_behind > -0.18 or abs(flank_lateral) > 0.44 or raw_vx < 0.08:
            if ball_dist < 0.82 and raw_vx < 0.12:
                vx = clamp(0.12 * raw_vx, -0.08, 0.14)
                vy = clamp(1.0 * raw_vy + 0.18 * b_y, -0.95, 0.95)
                w = clamp(math.radians(flank_err_deg) * 1.05, -0.9, 0.9)
                _set_behavior(
                    agent,
                    "body_push_own_flank_turn_clear",
                    ball_dist=round(ball_dist, 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    local_x=round(b_x, 2),
                    local_y=round(b_y, 2),
                    raw_vx=round(raw_vx, 2),
                    cmd_x=round(vx, 2),
                    cmd_y=round(vy, 2),
                )
                agent.cmd_vel(vx, vy, w)
                agent.move_head(math.inf, math.inf)
                return True
            setup_x = clamp(ball_x - 0.45 * flank_dx, -half_len + 0.28, -1.85)
            setup_y = clamp(ball_y - 0.45 * flank_dy, -half_wid + 0.28, half_wid - 0.28)
            _set_behavior(
                agent,
                "body_push_own_flank_setup",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                behind=round(flank_behind, 2),
                lateral=round(flank_lateral, 2),
            )
            _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.1)
            agent.move_head(math.inf, math.inf)
            return True
        vx = clamp(1.15 * raw_vx, 0.55, 1.0)
        vy = clamp(1.05 * raw_vy + 0.12 * b_y, -0.88, 0.88)
        w = clamp(
            math.radians(flank_err_deg) * 0.65 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.72, 0.72
        )
        _set_behavior(
            agent,
            "body_push_own_flank_diagonal_clear",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x < -half_len + 1.05 and abs(ball_y) > 1.35 and ball_dist < 1.12:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = -half_len + 2.7
        target_y = side * 0.08
        clear_dx = target_x - ball_x
        clear_dy = target_y - ball_y
        clear_norm = max(math.hypot(clear_dx, clear_dy), 1e-6)
        clear_dx /= clear_norm
        clear_dy /= clear_norm
        clear_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        clear_err_deg = normalize_angle_deg(clear_yaw_deg - yaw_deg)
        vx = clear_dx * math.cos(yaw_rad) + clear_dy * math.sin(yaw_rad)
        vy = -clear_dx * math.sin(yaw_rad) + clear_dy * math.cos(yaw_rad)
        if vx < 0.16:
            setup_x = clamp(ball_x - 0.48, -half_len + 0.18, -half_len + 0.95)
            setup_y = clamp(ball_y + side * 0.52, -half_wid + 0.35, half_wid - 0.35)
            _set_behavior(
                agent,
                "body_push_own_corner_reposition",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
            )
            _drive_to_pose(
                agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.11
            )
            agent.move_head(math.inf, math.inf)
            return True
        vx = clamp(1.25 * vx, 0.72, 1.0)
        vy = clamp(1.05 * vy - 0.18 * side + 0.15 * b_y, -0.9, 0.9)
        w = clamp(
            math.radians(clear_err_deg) * 0.42 + 0.18 * math.atan2(b_y, max(b_x, 0.04)), -0.52, 0.52
        )
        _set_behavior(
            agent,
            "body_push_own_corner_clear",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if (
        ball_x > goal_x - 0.85
        and goal_edge_y < abs(ball_y) < 2.35
        and ball_dist < 1.18
        and b_x > -0.36
    ):
        side = 1.0 if ball_y > 0.0 else -1.0
        if ball_x > goal_x - 0.42:
            sweep_yaw_deg = math.degrees(math.atan2(-ball_y, max(goal_x - ball_x, 0.06)))
            sweep_err_deg = normalize_angle_deg(sweep_yaw_deg - yaw_deg)
            vx = 0.06 if ball_x > goal_x - 0.14 else 0.16
            if b_x < 0.02:
                vx = 0.02
            vy = clamp(-1.28 * side + 0.20 * b_y, -1.0, 1.0)
            w = clamp(
                math.radians(sweep_err_deg) * 0.8 + 0.10 * math.atan2(b_y, max(b_x, 0.04)),
                -0.85,
                0.85,
            )
            _set_behavior(
                agent,
                "body_push_front_wide_sweep_in",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return True

        target_x = goal_x + 0.35
        target_y = 0.0
        crash_dx = target_x - ball_x
        crash_dy = target_y - ball_y
        crash_norm = max(math.hypot(crash_dx, crash_dy), 1e-6)
        crash_dx /= crash_norm
        crash_dy /= crash_norm
        raw_vx = crash_dx * math.cos(yaw_rad) + crash_dy * math.sin(yaw_rad)
        raw_vy = -crash_dx * math.sin(yaw_rad) + crash_dy * math.cos(yaw_rad)
        crash_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        crash_err_deg = normalize_angle_deg(crash_yaw_deg - yaw_deg)
        if raw_vx < 0.08:
            vx = clamp(0.28 * raw_vx, -0.18, 0.14)
            vy = clamp(1.05 * raw_vy + 0.22 * b_y, -0.95, 0.95)
            w = clamp(math.radians(crash_err_deg) * 1.0, -0.9, 0.9)
            _set_behavior(
                agent,
                "body_push_front_wide_turn_in",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                raw_vx=round(raw_vx, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return True
        vx = clamp(1.25 * raw_vx, 0.52, 1.0)
        vy = clamp(1.15 * raw_vy + 0.28 * b_y, -1.0, 1.0)
        w = clamp(
            math.radians(crash_err_deg) * 0.58 + 0.16 * math.atan2(b_y, max(b_x, 0.04)), -0.65, 0.65
        )
        _set_behavior(
            agent,
            "body_push_front_wide_crash",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 1.55 and abs(ball_y) > half_wid - 0.25 and ball_dist < 1.12:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x + 0.5
        target_y = side * 0.18
        end_dx = target_x - ball_x
        end_dy = target_y - ball_y
        end_norm = max(math.hypot(end_dx, end_dy), 1e-6)
        end_dx /= end_norm
        end_dy /= end_norm
        rel_x = pos[0] - ball_x
        rel_y = pos[1] - ball_y
        end_behind = rel_x * end_dx + rel_y * end_dy
        end_lateral = -rel_x * end_dy + rel_y * end_dx
        raw_vx = end_dx * math.cos(yaw_rad) + end_dy * math.sin(yaw_rad)
        raw_vy = -end_dx * math.sin(yaw_rad) + end_dy * math.cos(yaw_rad)
        end_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        end_err_deg = normalize_angle_deg(end_yaw_deg - yaw_deg)
        if end_behind > -0.16 or abs(end_lateral) > 0.42 or raw_vx < 0.08:
            if ball_dist < 0.78 and raw_vx < 0.12:
                vx = clamp(0.12 * raw_vx, -0.08, 0.12)
                vy = clamp(1.05 * raw_vy + 0.22 * b_y, -1.0, 1.0)
                w = clamp(math.radians(end_err_deg) * 1.05, -0.95, 0.95)
                _set_behavior(
                    agent,
                    "body_push_endline_turn_goal",
                    ball_dist=round(ball_dist, 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    local_x=round(b_x, 2),
                    local_y=round(b_y, 2),
                    raw_vx=round(raw_vx, 2),
                    cmd_x=round(vx, 2),
                    cmd_y=round(vy, 2),
                )
                agent.cmd_vel(vx, vy, w)
                agent.move_head(math.inf, math.inf)
                return True
            setup_x = clamp(ball_x - 0.44 * end_dx, goal_x - 1.65, goal_x - 0.28)
            setup_y = clamp(ball_y - 0.44 * end_dy, -half_wid + 0.08, half_wid - 0.08)
            _set_behavior(
                agent,
                "body_push_endline_setup",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                behind=round(end_behind, 2),
                lateral=round(end_lateral, 2),
            )
            _drive_to_pose(
                agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.08
            )
            agent.move_head(math.inf, math.inf)
            return True
        vx = clamp(1.25 * raw_vx, 0.62, 1.0)
        vy = clamp(1.05 * raw_vy + 0.16 * b_y, -0.95, 0.95)
        w = clamp(
            math.radians(end_err_deg) * 0.62 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.72, 0.72
        )
        _set_behavior(
            agent,
            "body_push_endline_goal_drive",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 0.28 and abs(ball_y) > 1.28 and ball_dist < 0.92 and b_x > -0.22:
        side = 1.0 if ball_y > 0.0 else -1.0
        sweep_yaw_deg = math.degrees(math.atan2(-ball_y, max(goal_x - ball_x, 0.08)))
        sweep_err_deg = normalize_angle_deg(sweep_yaw_deg - yaw_deg)
        vx = 0.08 if ball_x > goal_x - 0.08 else 0.16
        if b_x < 0.02:
            vx = 0.02
        vy = clamp(-1.15 * side + 0.22 * b_y, -1.0, 1.0)
        w = clamp(
            math.radians(sweep_err_deg) * 0.75 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.8, 0.8
        )
        _set_behavior(
            agent,
            "body_push_goal_line_sweep_in",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 0.65 and abs(ball_y) < 0.16 and ball_dist < 0.82 and b_x > -0.26:
        target_x = goal_x + 1.05
        center_yaw_deg = math.degrees(math.atan2(-pos[1] * 0.15, target_x - pos[0]))
        center_err_deg = normalize_angle_deg(center_yaw_deg - yaw_deg)
        vx = 1.0 if b_x > -0.08 else 0.68
        vy = clamp(0.18 * b_y - 0.20 * ball_y, -0.18, 0.18)
        w = clamp(
            math.radians(center_err_deg) * 0.32 + 0.08 * math.atan2(b_y, max(b_x, 0.04)),
            -0.32,
            0.32,
        )
        _set_behavior(
            agent,
            "body_push_goalmouth_center_crash",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if (
        ball_x > goal_x - 0.65
        and 0.12 < abs(ball_y) < 0.42
        and ball_dist < 0.72
        and b_x > -0.22
        and abs(b_y) > 0.30
    ):
        target_x = goal_x + 0.95
        target_y = 0.0
        align_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        align_err_deg = normalize_angle_deg(align_yaw_deg - yaw_deg)
        vx = 0.18 if b_x > 0.02 else 0.04
        vy = clamp(1.45 * b_y - 0.18 * ball_y, -0.95, 0.95)
        w = clamp(
            math.radians(align_err_deg) * 0.72 + 0.28 * math.atan2(b_y, max(b_x, 0.04)), -0.75, 0.75
        )
        _set_behavior(
            agent,
            "body_push_goalmouth_lateral_align",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if (
        ball_x > goal_x - 0.6
        and goal_edge_y < abs(ball_y) < 2.25
        and ball_dist < 1.08
        and b_x > -0.28
    ):
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x + 0.18
        target_y = side * 0.62
        rescue_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        rescue_err_deg = normalize_angle_deg(rescue_yaw_deg - yaw_deg)
        vx = 0.42 if b_x > -0.05 else 0.22
        vy = clamp(0.38 * b_y - 0.92 * ball_y, -0.85, 0.85)
        w = clamp(math.radians(rescue_err_deg) * 0.45, -0.5, 0.5)
        _set_behavior(
            agent,
            "body_push_goalmouth_rescue",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 1.0 and abs(ball_y) < 1.05 and ball_dist < 1.2 and b_x > -0.24:
        target_x = goal_x + 0.95
        target_y = clamp(ball_y * 0.04, -0.06, 0.06)
        line_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        line_err_deg = normalize_angle_deg(line_yaw_deg - yaw_deg)
        vx = 1.0 if b_x > -0.04 else 0.62
        vy = clamp(0.40 * b_y - 0.55 * ball_y, -0.72, 0.72)
        if abs(ball_y) > goal_inner_y:
            vx = 0.94 if b_x > -0.04 else 0.58
            vy = clamp(0.30 * b_y - 0.88 * ball_y, -0.9, 0.9)
        w = clamp(
            math.radians(line_err_deg) * 0.5 + 0.20 * math.atan2(b_y, max(b_x, 0.04)), -0.58, 0.58
        )
        _set_behavior(
            agent,
            "body_push_goal_line_crash",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 1.18 and abs(ball_y) < 0.48 and ball_dist < 1.18 and b_x > -0.18:
        target_x = goal_x + 1.05
        target_y = 0.0
        punch_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        punch_err_deg = normalize_angle_deg(punch_yaw_deg - yaw_deg)
        vx = 1.0
        vy = clamp(0.22 * b_y - 0.18 * ball_y, -0.18, 0.18)
        w = clamp(
            math.radians(punch_err_deg) * 0.22 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.22, 0.22
        )
        _set_behavior(
            agent,
            "body_push_goalmouth_punch",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > 1.25 and abs(ball_y) > half_wid - 0.18 and ball_dist < 1.08:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x + 0.2
        target_y = side * 0.42
        push_dx = target_x - ball_x
        push_dy = target_y - ball_y
        push_norm = max(math.hypot(push_dx, push_dy), 1e-6)
        push_dx /= push_norm
        push_dy /= push_norm
        rel_x = pos[0] - ball_x
        rel_y = pos[1] - ball_y
        push_behind = rel_x * push_dx + rel_y * push_dy
        push_lateral = -rel_x * push_dy + rel_y * push_dx
        raw_vx = push_dx * math.cos(yaw_rad) + push_dy * math.sin(yaw_rad)
        raw_vy = -push_dx * math.sin(yaw_rad) + push_dy * math.cos(yaw_rad)
        push_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        push_err_deg = normalize_angle_deg(push_yaw_deg - yaw_deg)

        if push_behind > -0.20 or abs(push_lateral) > 0.44 or ball_dist > 0.92:
            setup_x = clamp(ball_x - 0.50 * push_dx, -half_len + 0.65, goal_x - 0.4)
            setup_y = clamp(ball_y - 0.50 * push_dy, -half_wid + 0.05, half_wid - 0.05)
            if abs(pos[0] - setup_x) < 0.16 and abs(pos[1] - setup_y) < 0.16:
                if ball_dist < 0.34 and raw_vx > 0.35 and b_x > 0.02:
                    vx = clamp(0.72 * raw_vx, 0.32, 0.72)
                    vy = clamp(0.95 * raw_vy + 0.22 * b_y, -0.9, 0.9)
                    w = clamp(
                        math.radians(push_err_deg) * 0.62 + 0.10 * math.atan2(b_y, max(b_x, 0.04)),
                        -0.7,
                        0.7,
                    )
                    behavior = "body_push_extreme_sideline_near_drive"
                else:
                    vx = clamp(0.10 * raw_vx, -0.10, 0.14)
                    vy = clamp(0.95 * raw_vy + 0.25 * b_y, -0.95, 0.95)
                    w = clamp(math.radians(push_err_deg) * 1.15, -1.0, 1.0)
                    behavior = "body_push_extreme_sideline_turn"
                _set_behavior(
                    agent,
                    behavior,
                    ball_dist=round(ball_dist, 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    local_x=round(b_x, 2),
                    local_y=round(b_y, 2),
                    raw_vx=round(raw_vx, 2),
                    cmd_x=round(vx, 2),
                    cmd_y=round(vy, 2),
                )
                agent.cmd_vel(vx, vy, w)
                agent.move_head(math.inf, math.inf)
                return True
            _set_behavior(
                agent,
                "body_push_extreme_sideline_setup",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                behind=round(push_behind, 2),
                lateral=round(push_lateral, 2),
            )
            _drive_to_pose(
                agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.08
            )
            agent.move_head(math.inf, math.inf)
            return True

        if raw_vx < 0.14:
            vx = clamp(0.18 * raw_vx, -0.12, 0.12)
            vy = clamp(0.92 * raw_vy + 0.15 * b_y, -0.95, 0.95)
            w = clamp(math.radians(push_err_deg) * 1.05, -0.9, 0.9)
        else:
            vx = clamp(1.2 * raw_vx, 0.58, 1.0)
            vy = clamp(0.95 * raw_vy + 0.12 * b_y, -0.78, 0.78)
            w = clamp(
                math.radians(push_err_deg) * 0.58 + 0.12 * math.atan2(b_y, max(b_x, 0.04)),
                -0.65,
                0.65,
            )
        _set_behavior(
            agent,
            "body_push_extreme_sideline_diagonal",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if abs(ball_y) > 1.5 and ball_dist < 1.12 and b_x > -0.2:
        side = 1.0 if ball_y > 0.0 else -1.0
        if ball_x > 2.15 and abs(ball_y) > 2.05 and b_x > -0.24:
            target_x = clamp(ball_x + 0.12, 2.35, goal_x - 0.55)
            target_y = side * 0.72
            cut_dx = target_x - ball_x
            cut_dy = target_y - ball_y
            cut_norm = max(math.hypot(cut_dx, cut_dy), 1e-6)
            cut_dx /= cut_norm
            cut_dy /= cut_norm
            raw_vx = cut_dx * math.cos(yaw_rad) + cut_dy * math.sin(yaw_rad)
            raw_vy = -cut_dx * math.sin(yaw_rad) + cut_dy * math.cos(yaw_rad)
            cut_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
            cut_err_deg = normalize_angle_deg(cut_yaw_deg - yaw_deg)
            vx = clamp(0.72 * raw_vx, 0.10, 0.46)
            vy = clamp(1.22 * raw_vy + 0.18 * b_y, -1.0, 1.0)
            w = clamp(
                math.radians(cut_err_deg) * 0.75 + 0.18 * math.atan2(b_y, max(b_x, 0.04)),
                -0.82,
                0.82,
            )
            _set_behavior(
                agent,
                "body_push_attack_corner_cutback",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return True

        if ball_x > -2.25 and abs(ball_y) > half_wid - 0.35 and b_x > -0.18:
            if ball_x < 1.45:
                target_x = clamp(ball_x + 1.9, 0.45, 2.75)
                target_y = side * 0.72
                lane_dx = target_x - ball_x
                lane_dy = target_y - ball_y
                lane_norm = max(math.hypot(lane_dx, lane_dy), 1e-6)
                lane_dx /= lane_norm
                lane_dy /= lane_norm
                raw_vx = lane_dx * math.cos(yaw_rad) + lane_dy * math.sin(yaw_rad)
                raw_vy = -lane_dx * math.sin(yaw_rad) + lane_dy * math.cos(yaw_rad)
                lane_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
                lane_err_deg = normalize_angle_deg(lane_yaw_deg - yaw_deg)
                if raw_vx < 0.12:
                    vx = clamp(0.18 * raw_vx, -0.12, 0.14)
                    vy = clamp(0.92 * raw_vy + 0.16 * b_y, -0.95, 0.95)
                    w = clamp(math.radians(lane_err_deg) * 1.05, -0.95, 0.95)
                    behavior = "body_push_mid_sideline_turn_lane"
                else:
                    vx = clamp(1.08 * raw_vx, 0.48, 0.92)
                    vy = clamp(0.88 * raw_vy + 0.14 * b_y, -0.85, 0.85)
                    w = clamp(
                        math.radians(lane_err_deg) * 0.68 + 0.12 * math.atan2(b_y, max(b_x, 0.04)),
                        -0.68,
                        0.68,
                    )
                    behavior = "body_push_mid_sideline_fast_lane"
                _set_behavior(
                    agent,
                    behavior,
                    ball_dist=round(ball_dist, 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    local_x=round(b_x, 2),
                    local_y=round(b_y, 2),
                    raw_vx=round(raw_vx, 2),
                    cmd_x=round(vx, 2),
                    cmd_y=round(vy, 2),
                    target_x=round(target_x, 2),
                    target_y=round(target_y, 2),
                )
                agent.cmd_vel(vx, vy, w)
                agent.move_head(math.inf, math.inf)
                return True
            if ball_x > 1.6:
                vx = 0.48 if b_x > 0.08 else 0.28
                vy = clamp(-1.08 * side + 0.16 * b_y, -1.0, 1.0)
            else:
                vx = 0.82 if b_x > 0.08 else 0.58
                vy = clamp(-0.95 * side + 0.18 * b_y, -0.95, 0.95)
            w = clamp(-0.42 * side + 0.25 * math.atan2(b_y, max(b_x, 0.04)), -0.7, 0.7)
            _set_behavior(
                agent,
                "body_push_sideline_escape",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return True

        outside_gap = side * (pos[1] - ball_y)
        setup_gap = 0.08 if ball_x > 1.35 else 0.18
        if ball_x > 1.85 and ball_dist < 0.5 and b_x > -0.10:
            desired_x = 0.72 if ball_x < goal_x - 1.25 else 0.48
            desired_y = -1.18 * side
            desired_norm = max(math.hypot(desired_x, desired_y), 1e-6)
            desired_x /= desired_norm
            desired_y /= desired_norm
            vx = desired_x * math.cos(yaw_rad) + desired_y * math.sin(yaw_rad)
            vy = -desired_x * math.sin(yaw_rad) + desired_y * math.cos(yaw_rad)
            vx = clamp(1.08 * vx, 0.26, 0.92)
            vy = clamp(1.25 * vy - 0.08 * side + 0.12 * b_y, -1.0, 1.0)
            w = clamp(-0.46 * side + 0.18 * math.atan2(b_y, max(b_x, 0.04)), -0.82, 0.82)
            _set_behavior(
                agent,
                "body_push_attack_side_cut_in",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return True

        if ball_x > 1.0 and outside_gap < setup_gap:
            setup_x = clamp(ball_x - 0.12, -half_len + 0.65, goal_x - 0.45)
            if ball_x > 2.1:
                setup_y = clamp(ball_y - side * 0.42, -half_wid + 0.42, half_wid - 0.42)
            else:
                setup_y = clamp(ball_y + side * 0.36, -half_wid + 0.42, half_wid - 0.42)
            _set_behavior(
                agent,
                "body_push_attack_side_setup",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
            )
            _drive_to_pose(agent, setup_x, setup_y, face_x=ball_x, face_y=ball_y, stop_dist=0.13)
            agent.move_head(math.inf, math.inf)
            return True

        desired_x = 0.85 if ball_x < 0.0 else (0.68 if ball_x < 2.3 else 0.58)
        desired_y = -1.0 if ball_y > 0.0 else 1.0
        desired_norm = max(math.hypot(desired_x, desired_y), 1e-6)
        desired_x /= desired_norm
        desired_y /= desired_norm
        inward_yaw_deg = math.degrees(math.atan2(-ball_y, max(goal_x - ball_x, 0.4)))
        inward_err_deg = normalize_angle_deg(inward_yaw_deg - yaw_deg)
        raw_vx = desired_x * math.cos(yaw_rad) + desired_y * math.sin(yaw_rad)
        raw_vy = -desired_x * math.sin(yaw_rad) + desired_y * math.cos(yaw_rad)
        if raw_vx < 0.12:
            vx = clamp(0.25 * raw_vx, -0.22, 0.18)
            vy = clamp(0.95 * raw_vy + 0.12 * b_y, -0.85, 0.85)
            w = clamp(math.radians(inward_err_deg) * 1.05, -0.95, 0.95)
            _set_behavior(
                agent,
                "body_push_side_turn_clear",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                raw_vx=round(raw_vx, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return True
        vx = clamp(1.05 * raw_vx, 0.55, 0.95)
        vy = clamp(0.95 * raw_vy + 0.1 * b_y, -0.85, 0.85)
        w = clamp(math.radians(inward_err_deg) * 0.9, -0.8, 0.8)
        _set_behavior(
            agent,
            "body_push_side_clear",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    # Reposition behind the ball first. This is slower than impulse kicks, but it
    # uses only velocity commands and avoids dragging the ball toward the sideline.
    near_sideline = abs(ball_y) > half_wid - 0.85
    # Do not start a body push until the robot is both behind the ball and
    # roughly facing the goal. Otherwise the gait spends effort side-stepping
    # around the ball and the ball barely moves.
    needs_reposition = (
        behind_error > -0.16
        or abs(lateral_error) > 0.42
        or b_x < -0.05
        or abs(yaw_err_deg) > 28.0
        or abs(b_y) > 0.34
    )
    if near_sideline and ball_dist < 0.95 and b_x > -0.12:
        needs_reposition = False

    if -0.35 < ball_x < 1.25 and abs(ball_y) < 0.82 and ball_dist < 0.95 and b_x > -0.28:
        vx = 1.0 if b_x > -0.10 else 0.62
        vy = clamp(0.36 * b_y - 0.34 * ball_y, -0.38, 0.38)
        w = clamp(0.35 * math.atan2(b_y, max(b_x, 0.04)), -0.32, 0.32)
        _set_behavior(
            agent,
            "body_push_midfield_breakout",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if needs_reposition:
        _set_behavior(
            agent,
            "body_push_reposition",
            ball_dist=round(ball_dist, 3),
            target_x=round(behind_x, 2),
            target_y=round(behind_y, 2),
        )
        _drive_to_pose(agent, behind_x, behind_y, face_x=goal_x, face_y=goal_y, stop_dist=0.11)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > 0.65 and abs(ball_y) > 1.35:
        desired_x = 0.38
        desired_y = -1.0 if ball_y > 0.0 else 1.0
        desired_norm = max(math.hypot(desired_x, desired_y), 1e-6)
        desired_x /= desired_norm
        desired_y /= desired_norm
        vx = desired_x * math.cos(yaw_rad) + desired_y * math.sin(yaw_rad)
        vy = -desired_x * math.sin(yaw_rad) + desired_y * math.cos(yaw_rad)
        vx = clamp(0.95 * vx, 0.18, 0.72)
        vy = clamp(1.2 * vy + 0.25 * b_y, -1.0, 1.0)
        inward_yaw_deg = math.degrees(math.atan2(-ball_y, goal_x - ball_x))
        inward_err_deg = normalize_angle_deg(inward_yaw_deg - yaw_deg)
        w = clamp(math.radians(inward_err_deg) * 0.9, -0.8, 0.8)
        _set_behavior(
            agent,
            "body_push_center_recover",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 2.05 and 0.62 < abs(ball_y) < 1.18 and ball_dist < 0.98 and b_x > -0.18:
        desired_x = 0.18
        desired_y = -1.0 if ball_y > 0.0 else 1.0
        desired_norm = max(math.hypot(desired_x, desired_y), 1e-6)
        desired_x /= desired_norm
        desired_y /= desired_norm
        vx = desired_x * math.cos(yaw_rad) + desired_y * math.sin(yaw_rad)
        vy = -desired_x * math.sin(yaw_rad) + desired_y * math.cos(yaw_rad)
        vx = clamp(0.72 * vx, 0.08, 0.38)
        vy = clamp(1.15 * vy + 0.25 * b_y, -0.88, 0.88)
        inward_yaw_deg = math.degrees(math.atan2(-ball_y, max(goal_x - ball_x, 0.35)))
        inward_err_deg = normalize_angle_deg(inward_yaw_deg - yaw_deg)
        w = clamp(math.radians(inward_err_deg) * 0.75, -0.65, 0.65)
        _set_behavior(
            agent,
            "body_push_goalmouth_center",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 1.65 and abs(ball_y) < 0.36 and ball_dist < 0.98 and b_x > -0.08:
        target_x = goal_x + 0.9
        target_y = 0.0
        direct_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        direct_err_deg = normalize_angle_deg(direct_yaw_deg - yaw_deg)
        vx = 1.0 if b_x > -0.03 else 0.82
        vy = clamp(0.32 * b_y - 0.28 * ball_y, -0.22, 0.22)
        w = clamp(
            math.radians(direct_err_deg) * 0.32 + 0.14 * math.atan2(b_y, max(b_x, 0.04)),
            -0.28,
            0.28,
        )
        _set_behavior(
            agent,
            "body_push_goalmouth_direct",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > goal_x - 2.05 and abs(ball_y) < goal_inner_y and b_x > -0.04:
        target_x = goal_x + 0.75
        target_y = clamp(ball_y * 0.06, -0.06, 0.06)
        gdx = target_x - ball_x
        gdy = target_y - ball_y
        gnorm = max(math.hypot(gdx, gdy), 1e-6)
        gdx /= gnorm
        gdy /= gnorm
        vx = gdx * math.cos(yaw_rad) + gdy * math.sin(yaw_rad)
        vy = -gdx * math.sin(yaw_rad) + gdy * math.cos(yaw_rad)
        vx = clamp(1.25 * vx, 0.72, 1.0)
        lateral_limit = 0.34 if abs(ball_y) < 0.42 else 0.68
        vy = clamp(0.82 * vy + 0.28 * b_y, -lateral_limit, lateral_limit)
        w = clamp(0.42 * math.atan2(b_y, max(b_x, 0.04)), -0.36, 0.36)
        _set_behavior(
            agent,
            "body_push_goalmouth_charge",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > 1.05 and b_x > -0.06 and 0.24 <= abs(b_y) < 0.62 and ball_dist < 0.88:
        vx = 0.28 if b_x > 0.04 else 0.12
        vy = clamp(1.35 * b_y, -0.7, 0.7)
        w = clamp(0.65 * math.atan2(b_y, max(b_x, 0.04)), -0.55, 0.55)
        _set_behavior(
            agent,
            "body_push_contact_align",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > 1.05 and b_x > -0.06 and abs(b_y) < 0.24 and ball_dist < 0.86:
        center_pull = -0.22 * ball_y
        vx = 1.0 if b_x > 0.02 else 0.72
        vy = clamp(0.55 * b_y + center_pull, -0.4, 0.4)
        w = clamp(0.45 * math.atan2(b_y, max(b_x, 0.04)), -0.4, 0.4)
        if ball_x > goal_x - 1.4:
            vy = clamp(0.75 * b_y - 0.7 * ball_y, -0.75, 0.75)
            w = clamp(0.55 * math.atan2(b_y, max(b_x, 0.04)), -0.5, 0.5)

        _set_behavior(
            agent,
            "body_push_contact_charge",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    if ball_x > 1.25 and abs(ball_y) < half_wid - 0.75:
        center_pull = -0.28 * ball_y
        if ball_x > goal_x - 1.4:
            center_pull = -0.75 * ball_y
        vx = 1.0 if b_x > -0.03 else 0.45
        vy = clamp(1.15 * b_y + center_pull, -0.65, 0.65)
        w = clamp(0.75 * math.atan2(b_y, max(b_x, 0.03)), -0.55, 0.55)
        if ball_dist < 0.36 and b_x > -0.02:
            vx = 1.0
            vy = clamp(0.65 * b_y + center_pull, -0.45, 0.45)
            w *= 0.55

        _set_behavior(
            agent,
            "body_push_front_charge",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return True

    # Drive through the ball toward the goal center in robot coordinates.
    desired_x = dir_x
    desired_y = dir_y - 0.55 * ball_y
    desired_norm = max(math.hypot(desired_x, desired_y), 1e-6)
    desired_x /= desired_norm
    desired_y /= desired_norm
    vx = desired_x * math.cos(yaw_rad) + desired_y * math.sin(yaw_rad)
    vy = -desired_x * math.sin(yaw_rad) + desired_y * math.cos(yaw_rad)
    vx = clamp(1.15 * vx, 0.25, 1.0)
    vy = clamp(1.15 * vy + 0.45 * b_y, -0.9, 0.9)
    w = clamp(math.radians(yaw_err_deg) * 0.8, -0.75, 0.75)

    if near_sideline:
        vx = 0.75
        vy = clamp(-2.0 * ball_y + 0.35 * b_y, -1.0, 1.0)
        w = clamp(math.radians(yaw_err_deg) * 0.35, -0.45, 0.45)

    if ball_map[0] > goal_x - 1.1:
        vx = 1.0
        vy = clamp(1.2 * b_y - 0.75 * ball_map[1], -0.9, 0.9)
        w = clamp(math.radians(yaw_err_deg) * 1.2, -0.9, 0.9)

    _set_behavior(
        agent,
        "body_push_goal",
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_map[0], 2),
        ball_y=round(ball_map[1], 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)
    return True


def _drive_to_pose(
    agent, target_x, target_y, face_x=None, face_y=None, face_yaw_deg=None, stop_dist=0.12
):
    pos = agent.get_self_pos()
    yaw_deg = agent.get_self_yaw()
    dx = target_x - pos[0]
    dy = target_y - pos[1]
    dist = math.hypot(dx, dy)
    yaw_rad = math.radians(yaw_deg)

    rel_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
    rel_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)

    if face_yaw_deg is None:
        if face_x is not None and face_y is not None:
            face_yaw_deg = math.degrees(math.atan2(face_y - pos[1], face_x - pos[0]))
        else:
            face_yaw_deg = 0.0

    yaw_err_deg = normalize_angle_deg(face_yaw_deg - yaw_deg)
    vx = clamp(rel_x * 0.8, -0.75, 0.85)
    vy = clamp(rel_y * 1.0, -0.8, 0.8)
    w = clamp(math.radians(yaw_err_deg) * 1.2, -1.1, 1.1)

    if dist < stop_dist:
        vx *= dist / max(stop_dist, 1e-6)
        vy *= dist / max(stop_dist, 1e-6)
        if abs(yaw_err_deg) < 6.0:
            vx = 0.0
            vy = 0.0
            w = 0.0
        else:
            vx *= 0.25
            vy *= 0.25

    agent.cmd_vel(vx, vy, w)


def _boundary_recovery(agent, tracker):
    pos = agent.get_self_pos()
    if pos is None or len(pos) < 2:
        return False

    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    x, y = float(pos[0]), float(pos[1])
    soft_out_x = x < -half_len + 0.05 or x > half_len - 0.05
    soft_out_y = y < -half_wid + 0.05 or y > half_wid - 0.05
    hard_out_x = x < -half_len - 0.16 or x > half_len + 0.16
    hard_out_y = y < -half_wid - 0.16 or y > half_wid + 0.16
    far_out = abs(y) > half_wid + 0.65 or abs(x) > half_len + 0.55
    if not (soft_out_x or soft_out_y or hard_out_x or hard_out_y or far_out):
        agent.boundary_recovery_ts = 0.0
        return False

    now = time.time()
    agent.boundary_recovery_ts = now
    ball_map = _get_team_ball(agent, tracker)

    role_lane = {
        "striker": 0.15,
        "support": -0.15,
        "sweeper": -0.35,
        "keeper": -half_len + 0.72,
    }
    base_x = role_lane.get(agent.role, -0.15)
    if ball_map is not None and agent.role != "keeper":
        ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
        if ball_x < -0.7:
            base_x = clamp(ball_x + 1.15, -0.45, 1.15)
        elif ball_x > 1.6:
            base_x = clamp(ball_x - 1.05, 0.55, 2.25)
        else:
            base_x = clamp(ball_x + (0.7 if agent.role == "striker" else -0.4), -0.4, 1.8)
    target_x = clamp(base_x, -half_len + 0.85, half_len - 0.9)
    target_y = clamp(y, -half_wid + 0.78, half_wid - 0.78)
    if soft_out_y or hard_out_y or far_out:
        target_y = clamp(-0.62 if y > 0.0 else 0.62, -0.86, 0.86)
    if soft_out_x or hard_out_x or far_out:
        target_x = clamp(
            -half_len + 0.92 if x < -half_len else half_len - 0.92 if x > half_len else target_x,
            -half_len + 0.9,
            half_len - 0.9,
        )
    if agent.role == "keeper":
        target_x = -half_len + 0.72
        target_y = clamp(target_y, -0.72, 0.72)

    dx = target_x - x
    dy = target_y - y
    dist = math.hypot(dx, dy)
    yaw_deg = agent.get_self_yaw()
    yaw_rad = math.radians(yaw_deg)
    world_yaw_deg = math.degrees(math.atan2(dy, dx))
    yaw_err_deg = normalize_angle_deg(world_yaw_deg - yaw_deg)
    rel_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
    rel_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)

    # VLA-style safety affordance: returning in-bounds dominates ball intent.
    # Use holonomic world-vector control, not a turn-then-walk FSM, so a robot
    # that is already far outside cannot keep drifting while rotating.
    scale = min(1.0, dist / 1.2)
    vx = clamp(0.58 * rel_x, -0.72, 0.72)
    vy = clamp(0.72 * rel_y, -0.95, 0.95)
    if far_out:
        vx = clamp(0.82 * rel_x, -0.9, 0.9)
        vy = clamp(0.98 * rel_y, -1.0, 1.0)
    if dist > 0.35 and abs(vx) < 0.12 and abs(rel_x) > 0.12:
        vx = math.copysign(0.12, rel_x)
    if dist > 0.35 and abs(vy) < 0.18 and abs(rel_y) > 0.18:
        vy = math.copysign(0.18, rel_y)
    vx *= max(0.45, scale)
    vy *= max(0.45, scale)
    face_x = float(ball_map[0]) if ball_map is not None and not far_out else target_x
    face_y = float(ball_map[1]) if ball_map is not None and not far_out else target_y
    face_yaw_deg = math.degrees(math.atan2(face_y - y, face_x - x))
    face_err_deg = normalize_angle_deg(face_yaw_deg - yaw_deg)
    w = clamp(math.radians(face_err_deg) * (0.55 if far_out else 0.8), -0.9, 0.9)
    behavior = (
        "embodied_boundary_reentry"
        if far_out or hard_out_x or hard_out_y
        else "embodied_boundary_guardrail"
    )
    _set_behavior(
        agent,
        behavior,
        target_x=round(target_x, 2),
        target_y=round(target_y, 2),
        out_x=round(x, 2),
        out_y=round(y, 2),
        dist=round(dist, 2),
        yaw_err=round(yaw_err_deg, 1),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)
    return True


def _get_ready_pose(agent):
    half_len = agent.field_length / 2.0
    poses = {
        "striker": (0.15, -0.35, 0.0),
        "support": (-0.35, 0.35, 0.0),
        "sweeper": (-1.25, 0.0, 0.0),
        "keeper": (-half_len + 0.6, 0.0, 0.0),
    }
    return poses.get(agent.role, (-1.0, 0.0, 0.0))


def _get_setplay_ready_pose(agent, tracker):
    gc = agent.gamecontroller
    if gc.set_play_name != "SET_PLAY_KICK_OFF":
        return _get_ready_pose(agent)

    if gc.can_kick or gc.kicking_team == gc.team:
        half_len = agent.field_length / 2.0
        poses = {
            "striker": (-0.48, 0.12, 0.0),
            "support": (-1.15, -0.72, 0.0),
            "sweeper": (-1.25, 0.55, 0.0),
            "keeper": (-half_len + 0.6, 0.0, 0.0),
        }
        return poses.get(agent.role, _get_ready_pose(agent))

    half_len = agent.field_length / 2.0
    # Opponent kickoff READY can still include stale/rolling ball poses in sim.
    # Do not chase a moving ball here: fixed wide lanes avoid pushing it into our own goal.
    poses = {
        "striker": (-1.35, -1.35, 0.0),
        "support": (-1.35, 1.35, 0.0),
        "sweeper": (-2.55, 0.0, 0.0),
        "keeper": (-half_len + 0.6, 0.0, 0.0),
    }
    return poses.get(agent.role, _get_ready_pose(agent))


def _get_team_ball(agent, tracker):
    if agent.get_if_ball():
        return tuple(float(v) for v in agent.get_ball_pos_in_map())
    return tracker.get_ball_map()


def _field_phase(agent, ball_map):
    half_len = agent.field_length / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    if ball_x < -half_len + 1.6:
        return "defend_box"
    if ball_x < -0.5:
        return "defend"
    if ball_x < 1.3:
        return "midfield"
    if ball_x < half_len - 1.25:
        return "advance"
    if abs(ball_y) < 1.05:
        return "finish"
    return "wide_finish"


def _front_wide_recovery_target(agent, ball_map):
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    side = 1.0 if ball_y > 0.0 else -1.0

    # For a front-corner ball, the useful position is outside-and-slightly-behind
    # the ball so the next body contact sends it back into the goal corridor.
    target_x = clamp(ball_x - 0.32, half_len - 1.55, half_len - 0.28)
    target_y = clamp(ball_y + side * 0.48, -half_wid + 0.22, half_wid - 0.22)
    face_x = half_len + 0.35
    face_y = clamp(ball_y * 0.12, -0.22, 0.22)
    return target_x, target_y, face_x, face_y


def _own_kickoff_opening(agent, tracker, ball_map):
    gc = agent.gamecontroller
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    kickoff_lane_ok = -0.30 < ball_x < 0.72 and abs(ball_y) < 0.52
    if (
        gc.set_play_name != "SET_PLAY_KICK_OFF"
        or not gc.can_kick
        or gc.kicking_team != gc.team
        or not kickoff_lane_ok
    ):
        return False

    if not agent.get_if_ball():
        _set_behavior(agent, "kickoff_find_ball")
        agent.state_machine_runners["find_ball"]()
        return True

    goal_x = agent.field_length / 2.0
    half_len = agent.field_length / 2.0
    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    pos = agent.get_self_pos()

    drive_ready = (
        b_x > 0.22
        and abs(b_y) < 0.18
        and ball_dist < 0.82
        and pos[0] < ball_x - 0.14
        and abs(normalize_angle_deg(agent.get_self_yaw())) < 22.0
    )
    if not drive_ready:
        setup_x = clamp(ball_x - 0.72, -half_len + 0.8, -0.26)
        setup_y = clamp(ball_y - 0.10 * b_y, -0.08, 0.08)
        _set_behavior(
            agent,
            "kickoff_center_setup",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=goal_x, face_y=0.0, stop_dist=0.08)
        agent.move_head(math.inf, math.inf)
        return True

    yaw_err_deg = normalize_angle_deg(0.0 - agent.get_self_yaw())
    vx = 0.94
    vy = clamp(0.04 * b_y - 0.10 * ball_y, -0.09, 0.09)
    w = clamp(
        math.radians(yaw_err_deg) * 0.14 + 0.03 * math.atan2(b_y, max(b_x, 0.04)), -0.16, 0.16
    )
    _set_behavior(
        agent,
        "kickoff_center_drive",
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        yaw_err=round(yaw_err_deg, 1),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)
    return True


def _primary_ball_handler(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        _set_behavior(agent, "find_ball_no_team_ball")
        agent.state_machine_runners["find_ball"]()
        return

    goal_x = agent.field_length / 2.0
    own_goal_x = -agent.field_length / 2.0
    shooting_zone_x = goal_x - 1.2

    if _own_kickoff_opening(agent, tracker, ball_map):
        return

    if agent.get_if_ball() and agent.get_ball_distance() < 0.32 and ball_map[0] > shooting_zone_x:
        if time.time() - agent.last_shot_attempt_ts > 1.2:
            tracker.recorder.log(
                "shot_attempt",
                robot=int(getattr(agent, "local_id", agent.get_config().get("id", 0))),
                role=agent.role,
                ball_x=round(ball_map[0], 3),
                ball_y=round(ball_map[1], 3),
            )
            agent.last_shot_attempt_ts = time.time()

    if agent.role == "keeper":
        if agent.get_if_ball() and ball_map[0] < own_goal_x + 1.65 and abs(ball_map[1]) < 1.18:
            _set_behavior(
                agent, "keeper_primary_clearance", ball_dist=round(agent.get_ball_distance(), 3)
            )
            _keeper_clear_ball(agent, tracker, ball_map)
            return
        _keeper_anchor(agent, tracker)
        return

    if _finish_attack(agent, tracker, ball_map):
        _set_behavior(agent, "finish_attack", ball_dist=round(agent.get_ball_distance(), 3))
        return

    if (
        agent.get_if_ball()
        and ball_map[0] > goal_x - 1.65
        and abs(ball_map[1]) > 1.08
        and agent.get_ball_distance() > 0.78
    ):
        side = 1.0 if ball_map[1] > 0.0 else -1.0
        if ball_map[0] > goal_x - 1.75 and agent.get_ball_distance() > 1.25:
            target_x, target_y, face_x, face_y = _front_wide_recovery_target(agent, ball_map)
            _set_behavior(
                agent,
                "attack_deep_wide_outside_recover",
                ball_dist=round(agent.get_ball_distance(), 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
            )
            _drive_to_pose(agent, target_x, target_y, face_x=face_x, face_y=face_y, stop_dist=0.12)
            return
        if (
            ball_map[0] > goal_x - 1.15
            and abs(ball_map[1]) > 1.85
            and agent.get_ball_distance() < 3.45
        ):
            target_x, target_y, face_x, face_y = _front_wide_recovery_target(agent, ball_map)
            if agent.get_ball_distance() > 1.05:
                _set_behavior(
                    agent,
                    "attack_deep_wide_outside_press",
                    ball_dist=round(agent.get_ball_distance(), 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    target_x=round(target_x, 2),
                    target_y=round(target_y, 2),
                )
                _drive_to_pose(
                    agent, target_x, target_y, face_x=face_x, face_y=face_y, stop_dist=0.12
                )
                return
            _set_behavior(
                agent,
                "attack_deep_wide_goal_press",
                ball_dist=round(agent.get_ball_distance(), 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
            )
            if _aggressive_pursuit(agent, target_distance=0.34):
                return
        if (
            ball_map[0] > goal_x - 0.55
            and abs(ball_map[1]) < 2.18
            and agent.get_ball_distance() < 3.8
        ):
            _set_behavior(
                agent,
                "attack_deep_wide_goal_press",
                ball_dist=round(agent.get_ball_distance(), 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
            )
            if _aggressive_pursuit(agent, target_distance=0.32):
                return
        if ball_map[0] > goal_x - 0.85 and agent.get_ball_distance() < 1.18:
            _set_behavior(
                agent,
                "attack_wide_goal_press",
                ball_dist=round(agent.get_ball_distance(), 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
            )
            if _aggressive_pursuit(agent, target_distance=0.58):
                return
        setup_x, setup_y, face_x, face_y = _front_wide_recovery_target(agent, ball_map)
        _set_behavior(
            agent,
            "attack_wide_outside_setup",
            ball_dist=round(agent.get_ball_distance(), 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=face_x, face_y=face_y, stop_dist=0.14)
        return

    if agent.get_if_ball() and agent.get_ball_distance() > 0.9:
        _set_behavior(agent, "aggressive_pursuit", ball_dist=round(agent.get_ball_distance(), 3))
        if _aggressive_pursuit(agent, target_distance=0.48):
            return

    if agent.get_if_ball() and agent.get_ball_distance() < 0.75 and ball_map[0] > goal_x - 2.2:
        _set_behavior(agent, "adv_dribble_finish", ball_dist=round(agent.get_ball_distance(), 3))
        agent.state_machine_runners["adv_dribble"]()
        return

    _run_find_or_chase(agent, chase_distance=0.48)


def _support_anchor(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        rx, ry, ryaw = _get_ready_pose(agent)
        _set_behavior(agent, "support_reset")
        _drive_to_pose(agent, rx, ry, face_yaw_deg=ryaw)
        return

    half_len = agent.field_length / 2.0
    target_x = clamp(ball_map[0] - 0.45, -half_len + 1.0, 2.2)
    target_y = clamp(ball_map[1] * 0.45, -1.15, 1.15)

    if (
        agent.gamecontroller.set_play_name == "SET_PLAY_KICK_OFF"
        and agent.gamecontroller.can_kick
        and ball_map[0] < 0.7
        and abs(ball_map[1]) < 0.7
    ):
        if agent.role == "striker":
            target_x = clamp(ball_map[0] + 1.15, 0.85, 1.55)
            target_y = -0.42 if ball_map[1] >= 0.0 else 0.42
        else:
            target_x = clamp(ball_map[0] - 1.05, -half_len + 0.9, -0.55)
            target_y = -0.85 if ball_map[1] >= 0.0 else 0.85
        _set_behavior(
            agent,
            "support_kickoff_second_ball",
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(
            agent, target_x, target_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.16
        )
        return

    if ball_map[0] < -0.6:
        target_x = clamp(ball_map[0] - 0.45, -half_len + 0.9, 0.5)
        target_y = clamp(ball_map[1], -1.6, 1.6)

    if agent.role == "striker" and ball_map[0] > 0.45:
        if ball_map[0] > 1.65 and abs(ball_map[1]) > 1.25:
            side = 1.0 if ball_map[1] > 0.0 else -1.0
            receive_x = clamp(max(ball_map[0] + 1.55, 3.55), 2.75, half_len - 0.38)
            receive_y = clamp(side * 0.28, -0.42, 0.42)
        else:
            receive_x = clamp(max(ball_map[0] + 1.9, 3.15), 1.9, 3.9)
            receive_y = clamp(ball_map[1] * 0.35, -0.65, 0.65)
        if agent.get_if_ball() and agent.get_ball_distance() < 1.15:
            _set_behavior(agent, "striker_takeover", ball_dist=round(agent.get_ball_distance(), 3))
            if _finish_attack(agent, tracker, ball_map):
                return
            _aggressive_pursuit(agent, target_distance=0.72)
            return
        _set_behavior(
            agent,
            "striker_receive_lane",
            target_x=round(receive_x, 2),
            target_y=round(receive_y, 2),
        )
        _drive_to_pose(
            agent, receive_x, receive_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.16
        )
        return

    if (
        agent.get_if_ball()
        and agent.get_ball_distance() < 1.02
        and ball_map[0] < -2.55
        and abs(ball_map[1]) > 1.65
    ):
        _set_behavior(
            agent, "support_deep_corner_clear", ball_dist=round(agent.get_ball_distance(), 3)
        )
        if _finish_attack(agent, tracker, ball_map):
            return
        _aggressive_pursuit(agent, target_distance=0.5)
        return

    if agent.get_if_ball() and agent.get_ball_distance() < 0.52 and ball_map[0] < -0.25:
        _set_behavior(agent, "support_emergency_press")
        if agent.get_ball_distance() > 0.42:
            _aggressive_pursuit(agent, target_distance=0.42)
        else:
            _drive_to_pose(
                agent,
                clamp(ball_map[0] - 0.45, -half_len + 0.8, 0.1),
                clamp(ball_map[1] * 0.35, -0.8, 0.8),
                face_x=ball_map[0],
                face_y=ball_map[1],
                stop_dist=0.14,
            )
        return

    if (
        ENABLE_SUPPORT_CO_PUSH
        and agent.get_if_ball()
        and -0.2 < ball_map[0] < 0.65
        and abs(ball_map[1]) < 1.45
        and agent.get_ball_distance() < 2.2
    ):
        if abs(ball_map[1]) < 0.75 and agent.get_ball_distance() < 1.35:
            _set_behavior(agent, "support_co_push", ball_dist=round(agent.get_ball_distance(), 3))
            if agent.get_ball_distance() > 0.62:
                _aggressive_pursuit(agent, target_distance=0.62)
            else:
                _drive_to_pose(
                    agent,
                    clamp(ball_map[0] - 0.55, -half_len + 0.9, 0.35),
                    clamp(ball_map[1] * 0.25, -0.45, 0.45),
                    face_x=ball_map[0],
                    face_y=ball_map[1],
                    stop_dist=0.16,
                )
            return
        _set_behavior(agent, "support_front_press")
        if agent.get_ball_distance() > 0.72 and _aggressive_pursuit(agent, target_distance=0.72):
            return

    if (
        not ENABLE_SUPPORT_CO_PUSH
        and agent.get_if_ball()
        and -0.2 < ball_map[0] < 0.65
        and abs(ball_map[1]) < 1.45
        and agent.get_ball_distance() < 2.2
    ):
        _set_behavior(agent, "support_front_press")
        if agent.get_ball_distance() > 0.72 and _aggressive_pursuit(agent, target_distance=0.72):
            return

    if ball_map[0] > 0.7:
        target_x = clamp(ball_map[0] + 0.2, 0.0, 3.35)
        side_bias = -0.55 if ball_map[1] >= 0 else 0.55
        target_y = clamp(ball_map[1] * 0.2 + side_bias, -1.1, 1.1)

    if ball_map[0] > 1.2:
        phase = _field_phase(agent, ball_map)
        if agent.role == "support" and phase in ("advance", "finish", "wide_finish"):
            score_margin = int(agent.gamecontroller.score) - int(
                agent.gamecontroller.opponent_score
            )
            if score_margin > 0:
                target_x = clamp(ball_map[0] - 1.25, 0.45, 1.85)
                target_y = clamp(ball_map[1] * 0.18, -0.65, 0.65)
                if abs(ball_map[1]) > 1.1:
                    target_y = clamp(-0.42 if ball_map[1] > 0.0 else 0.42, -0.65, 0.65)
                _set_behavior(
                    agent,
                    "support_rest_defense",
                    target_x=round(target_x, 2),
                    target_y=round(target_y, 2),
                    score_margin=score_margin,
                )
                _drive_to_pose(
                    agent,
                    target_x,
                    target_y,
                    face_x=ball_map[0],
                    face_y=ball_map[1],
                    stop_dist=0.15,
                )
                return
            if (
                phase == "advance"
                and agent.get_if_ball()
                and agent.get_ball_distance() < 1.25
                and abs(ball_map[1]) < 1.35
            ):
                b_x, b_y = [float(v) for v in agent.get_ball_pos()]
                ball_dist = math.hypot(b_x, b_y)
                pos = agent.get_self_pos()
                goal_x = half_len
                target_x = clamp(goal_x + 0.25, half_len - 0.1, half_len + 0.35)
                target_y = clamp(ball_map[1] * 0.12, -0.28, 0.28)
                drive_dx = target_x - ball_map[0]
                drive_dy = target_y - ball_map[1]
                drive_norm = max(math.hypot(drive_dx, drive_dy), 1e-6)
                drive_dx /= drive_norm
                drive_dy /= drive_norm
                rel_x = pos[0] - ball_map[0]
                rel_y = pos[1] - ball_map[1]
                behind = rel_x * drive_dx + rel_y * drive_dy
                lateral = -rel_x * drive_dy + rel_y * drive_dx
                ready = (
                    ball_dist < 0.78
                    and b_x > 0.02
                    and abs(b_y) < 0.58
                    and behind < -0.04
                    and abs(lateral) < 0.7
                )
                if not ready:
                    setup_x = clamp(ball_map[0] - 0.48 * drive_dx, 0.65, half_len - 1.0)
                    setup_y = clamp(ball_map[1] - 0.48 * drive_dy, -1.05, 1.05)
                    _set_behavior(
                        agent,
                        "support_advance_behind_setup",
                        ball_dist=round(ball_dist, 3),
                        ball_x=round(ball_map[0], 2),
                        ball_y=round(ball_map[1], 2),
                        local_x=round(b_x, 2),
                        local_y=round(b_y, 2),
                        target_x=round(setup_x, 2),
                        target_y=round(setup_y, 2),
                        behind=round(behind, 2),
                        lateral=round(lateral, 2),
                    )
                    _drive_to_pose(
                        agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.1
                    )
                    return
                raw_vx, raw_vy = _map_vector_to_robot(agent, drive_dx, drive_dy)
                drive_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
                drive_err_deg = normalize_angle_deg(drive_yaw_deg - agent.get_self_yaw())
                vx = clamp(1.08 * raw_vx, 0.42, 0.92)
                vy = clamp(0.72 * raw_vy + 0.12 * b_y - 0.18 * ball_map[1], -0.62, 0.62)
                w = clamp(
                    math.radians(drive_err_deg) * 0.45 + 0.10 * math.atan2(b_y, max(b_x, 0.04)),
                    -0.52,
                    0.52,
                )
                _set_behavior(
                    agent,
                    "support_advance_goal_drive",
                    ball_dist=round(ball_dist, 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    local_x=round(b_x, 2),
                    local_y=round(b_y, 2),
                    raw_vx=round(raw_vx, 2),
                    raw_vy=round(raw_vy, 2),
                    cmd_x=round(vx, 2),
                    cmd_y=round(vy, 2),
                )
                agent.cmd_vel(vx, vy, w)
                agent.move_head(math.inf, math.inf)
                return
        if phase == "wide_finish":
            side = 1.0 if ball_map[1] > 0.0 else -1.0
            if agent.get_if_ball() and agent.get_ball_distance() < 1.08:
                pos = agent.get_self_pos()
                if pos[0] < ball_map[0] - 0.15 and side * pos[1] > side * ball_map[1] + 0.08:
                    _set_behavior(
                        agent,
                        "support_wide_finish_touch",
                        ball_dist=round(agent.get_ball_distance(), 3),
                    )
                    if _finish_attack(agent, tracker, ball_map):
                        return
                    _aggressive_pursuit(agent, target_distance=0.54)
                    return
            target_x = clamp(ball_map[0] - 0.25, 3.15, half_len - 0.28)
            target_y = clamp(side * 0.42, -0.54, 0.54)
            if ball_map[0] > half_len - 0.7:
                target_x = half_len - 0.45
                target_y = clamp(side * 0.34, -0.46, 0.46)
            _set_behavior(
                agent,
                "support_wide_finish_second_ball",
                ball_dist=round(agent.get_ball_distance(), 3) if agent.get_if_ball() else None,
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
            )
            _drive_to_pose(
                agent, target_x, target_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.12
            )
            return

        if ball_map[0] > half_len - 1.0 and agent.get_if_ball() and agent.get_ball_distance() < 2.4:
            if agent.get_ball_distance() < 1.15:
                _set_behavior(
                    agent, "support_deep_goal_crash", ball_dist=round(agent.get_ball_distance(), 3)
                )
                if _finish_attack(agent, tracker, ball_map):
                    return
            target_x = clamp(ball_map[0] - 0.35, 3.0, half_len - 0.35)
            target_y = clamp(ball_map[1] * 0.10, -0.24, 0.24)
            _set_behavior(
                agent,
                "support_deep_goalmouth_press",
                ball_dist=round(agent.get_ball_distance(), 3),
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
            )
            _drive_to_pose(
                agent, target_x, target_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.12
            )
            return

        if (
            ENABLE_SUPPORT_CO_PUSH
            and agent.get_if_ball()
            and abs(ball_map[1]) < 0.95
            and agent.get_ball_distance() < 2.6
        ):
            _set_behavior(agent, "support_co_push", ball_dist=round(agent.get_ball_distance(), 3))
            if agent.get_ball_distance() > 0.65:
                _aggressive_pursuit(agent, target_distance=0.65)
                return
            if _finish_attack(agent, tracker, ball_map):
                return

        if abs(ball_map[1]) >= 0.95:
            if ball_map[0] > half_len - 0.75:
                target_x = clamp(ball_map[0] - 0.35, 3.15, half_len - 0.42)
                target_y = clamp(ball_map[1] * 0.22, -0.52, 0.52)
            elif ball_map[0] > 1.75:
                target_x = clamp(ball_map[0] + 0.78, 2.45, half_len - 0.65)
                target_y = clamp(ball_map[1] * 0.14, -0.36, 0.36)
            else:
                target_x = clamp(ball_map[0] + 0.25, 0.45, 3.05)
                target_y = clamp(ball_map[1] * 0.22, -0.48, 0.48)
            _set_behavior(
                agent,
                "support_inside_lane",
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
            )
            _drive_to_pose(
                agent, target_x, target_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.14
            )
            return

        target_x = clamp(ball_map[0] + 0.15, 0.35, 3.1)
        target_y = clamp(ball_map[1] * 0.32, -0.72, 0.72)
        if abs(ball_map[1]) > 0.8:
            target_y = clamp(ball_map[1] * 0.12, -0.36, 0.36)
        if agent.get_if_ball() and agent.get_ball_distance() < 0.92:
            _set_behavior(
                agent, "support_second_ball_drive", ball_dist=round(agent.get_ball_distance(), 3)
            )
            if _finish_attack(agent, tracker, ball_map):
                return
            _aggressive_pursuit(agent, target_distance=0.58)
            return
        _set_behavior(
            agent,
            "support_second_ball_hold",
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(
            agent, target_x, target_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.16
        )
        return

    _set_behavior(agent, "support_cover", target_x=round(target_x, 2), target_y=round(target_y, 2))
    _drive_to_pose(
        agent, target_x, target_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.14
    )


def _sweeper_anchor(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        tx, ty, tyaw = _get_ready_pose(agent)
        _set_behavior(agent, "sweeper_reset")
        _drive_to_pose(agent, tx, ty, face_yaw_deg=tyaw, stop_dist=0.12)
        return

    half_len = agent.field_length / 2.0
    if agent.get_if_ball() and agent.get_ball_distance() < 0.9:
        if ball_map[0] < 0.1:
            _set_behavior(
                agent, "sweeper_clear_or_press", ball_dist=round(agent.get_ball_distance(), 3)
            )
            if _finish_attack(agent, tracker, ball_map):
                return
            _aggressive_pursuit(agent, target_distance=0.48)
            return
        if ball_map[0] > 0.55:
            _set_behavior(
                agent, "sweeper_second_striker", ball_dist=round(agent.get_ball_distance(), 3)
            )
            if _finish_attack(agent, tracker, ball_map):
                return
            _aggressive_pursuit(agent, target_distance=0.55)
            return

    target_x = clamp(ball_map[0] - 0.85, -half_len + 0.75, 2.0)
    target_y = clamp(ball_map[1] * 0.35, -1.05, 1.05)
    if ball_map[0] < -0.8:
        target_x = clamp(ball_map[0] - 0.25, -half_len + 0.6, -0.4)
        target_y = clamp(ball_map[1] * 0.75, -1.35, 1.35)

    _set_behavior(agent, "sweeper_screen", target_x=round(target_x, 2), target_y=round(target_y, 2))
    _drive_to_pose(
        agent, target_x, target_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.12
    )


def _keeper_clear_ball(agent, tracker, ball_map):
    if not agent.get_if_ball():
        _set_behavior(agent, "keeper_find_clear")
        agent.state_machine_runners["find_ball"]()
        return

    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    own_goal_x = -agent.field_length / 2.0
    if ball_map[0] < own_goal_x + 0.58 and abs(ball_map[1]) < 1.18:
        field_target_x = own_goal_x + 1.95
        field_target_y = 0.0
        if ball_map[0] < own_goal_x + 0.28 and ball_dist < 1.15 and b_x > -0.28:
            panic_dx = field_target_x - ball_map[0]
            panic_dy = field_target_y - ball_map[1]
            panic_norm = max(math.hypot(panic_dx, panic_dy), 1e-6)
            panic_dx /= panic_norm
            panic_dy /= panic_norm
            panic_vx, panic_vy = _map_vector_to_robot(agent, panic_dx, panic_dy)
            panic_yaw_deg = math.degrees(
                math.atan2(
                    field_target_y - agent.get_self_pos()[1],
                    field_target_x - agent.get_self_pos()[0],
                )
            )
            panic_err_deg = normalize_angle_deg(panic_yaw_deg - agent.get_self_yaw())
            vx = clamp(0.92 * panic_vx + 0.18 * max(b_x, -0.15), 0.16, 0.72)
            vy = clamp(0.42 * panic_vy + 0.10 * b_y - 0.52 * ball_map[1], -0.34, 0.34)
            w = clamp(
                math.radians(panic_err_deg) * 0.48 + 0.08 * math.atan2(b_y, max(abs(b_x), 0.06)),
                -0.58,
                0.58,
            )
            _set_behavior(
                agent,
                "keeper_goal_line_panic_center_clear",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return
        if ball_dist > 1.05:
            target_x = clamp(ball_map[0] + 0.68, own_goal_x + 0.82, own_goal_x + 1.22)
            target_y = clamp(ball_map[1] * 0.62, -0.72, 0.72)
            block_vx, block_vy = _map_vector_to_robot(
                agent, target_x - agent.get_self_pos()[0], target_y - agent.get_self_pos()[1]
            )
            face_yaw_deg = math.degrees(
                math.atan2(
                    field_target_y - agent.get_self_pos()[1],
                    field_target_x - agent.get_self_pos()[0],
                )
            )
            yaw_err_deg = normalize_angle_deg(face_yaw_deg - agent.get_self_yaw())
            vx = clamp(1.25 * block_vx, -0.28, 0.88)
            if ball_map[0] < own_goal_x + 0.62 and block_vx < -0.08:
                vx = 0.02
            vy = clamp(1.25 * block_vy, -0.72, 0.72)
            w = clamp(math.radians(yaw_err_deg) * 1.1, -1.1, 1.1)
            _set_behavior(
                agent,
                "keeper_goal_line_close_block",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return

        yaw_rad = math.radians(agent.get_self_yaw())
        clear_dx = field_target_x - ball_map[0]
        clear_dy = field_target_y - ball_map[1]
        clear_norm = max(math.hypot(clear_dx, clear_dy), 1e-6)
        clear_dx /= clear_norm
        clear_dy /= clear_norm
        raw_vx = clear_dx * math.cos(yaw_rad) + clear_dy * math.sin(yaw_rad)
        raw_vy = -clear_dx * math.sin(yaw_rad) + clear_dy * math.cos(yaw_rad)
        if b_x < -0.18 or abs(b_y) > 0.42 or raw_vx < 0.08:
            lane_vx, lane_vy = _map_vector_to_robot(agent, clear_dx, clear_dy)
            if raw_vx < -0.12 or lane_vx < -0.08:
                target_x = own_goal_x + 0.52
                target_y = clamp(ball_map[1] * 0.72, -0.72, 0.72)
                _set_behavior(
                    agent,
                    "keeper_goal_line_realign_guard",
                    ball_dist=round(ball_dist, 3),
                    ball_x=round(ball_map[0], 2),
                    ball_y=round(ball_map[1], 2),
                    local_x=round(b_x, 2),
                    local_y=round(b_y, 2),
                    raw_vx=round(raw_vx, 2),
                    target_x=round(target_x, 2),
                    target_y=round(target_y, 2),
                )
                _drive_to_pose(
                    agent,
                    target_x,
                    target_y,
                    face_x=field_target_x,
                    face_y=field_target_y,
                    stop_dist=0.08,
                )
                agent.move_head(math.inf, math.inf)
                return
            vx = clamp(1.18 * lane_vx + 0.20 * max(b_x, -0.2), 0.08, 0.86)
            vy = clamp(0.62 * lane_vy + 0.14 * b_y - 0.36 * ball_map[1], -0.52, 0.52)
            w = clamp(
                0.65 * math.atan2(b_y, max(abs(b_x), 0.08)) * (1.0 if b_x >= 0.0 else -1.0),
                -0.75,
                0.75,
            )
            _set_behavior(
                agent,
                "keeper_goal_line_vector_clear",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                raw_vx=round(raw_vx, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return

        vx = clamp(1.2 * raw_vx, 0.58, 1.0)
        vy = clamp(0.52 * raw_vy + 0.20 * b_y - 0.42 * ball_map[1], -0.48, 0.48)
        w = clamp(0.32 * math.atan2(b_y, max(b_x, 0.04)), -0.35, 0.35)
        _set_behavior(
            agent,
            "keeper_goal_line_direct_clear",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    if ball_map[0] < own_goal_x + 0.95 and abs(ball_map[1]) > 0.72:
        field_target_x = own_goal_x + 1.65
        side = -1.0 if ball_map[1] > 0.0 else 1.0
        field_target_y = clamp(ball_map[1] * 0.12 + side * 0.42, -0.78, 0.78)
        if ball_dist > 0.58:
            target_x = clamp(ball_map[0] + 0.55, own_goal_x + 0.72, own_goal_x + 1.45)
            target_y = clamp(ball_map[1] * 0.48, -0.72, 0.72)
            _set_behavior(
                agent,
                "keeper_wide_goal_line_block",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
            )
            _drive_to_pose(
                agent,
                target_x,
                target_y,
                face_x=field_target_x,
                face_y=field_target_y,
                stop_dist=0.12,
            )
            return

        yaw_rad = math.radians(agent.get_self_yaw())
        clear_dx = field_target_x - ball_map[0]
        clear_dy = field_target_y - ball_map[1]
        clear_norm = max(math.hypot(clear_dx, clear_dy), 1e-6)
        clear_dx /= clear_norm
        clear_dy /= clear_norm
        raw_vx = clear_dx * math.cos(yaw_rad) + clear_dy * math.sin(yaw_rad)
        raw_vy = -clear_dx * math.sin(yaw_rad) + clear_dy * math.cos(yaw_rad)
        if raw_vx < 0.18 or b_x < -0.18:
            _set_behavior(
                agent,
                "keeper_wide_goal_line_turn_out",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                raw_vx=round(raw_vx, 2),
            )
            agent.cmd_vel(
                0.02,
                clamp(0.8 * raw_vy + 0.35 * b_y, -0.68, 0.68),
                clamp(0.9 * math.atan2(b_y, max(b_x, 0.04)), -0.85, 0.85),
            )
            agent.move_head(math.inf, math.inf)
            return
        vx = clamp(1.1 * raw_vx, 0.42, 0.88)
        vy = clamp(0.75 * raw_vy + 0.18 * b_y, -0.55, 0.55)
        w = clamp(0.45 * math.atan2(b_y, max(b_x, 0.04)), -0.45, 0.45)
        _set_behavior(
            agent,
            "keeper_wide_goal_line_clear",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    if ball_map[0] < own_goal_x + 0.12 and abs(ball_map[1]) < 1.35:
        field_target_x = own_goal_x + 1.45
        side = -1.0 if ball_map[1] > 0.0 else 1.0
        yaw_rad = math.radians(agent.get_self_yaw())
        clear_dx = field_target_x - agent.get_self_pos()[0]
        clear_dy = clamp(ball_map[1] * 0.15 + side * 0.42, -0.9, 0.9) - agent.get_self_pos()[1]
        vx = clear_dx * math.cos(yaw_rad) + clear_dy * math.sin(yaw_rad)
        vy = -clear_dx * math.sin(yaw_rad) + clear_dy * math.cos(yaw_rad)
        if vx < 0.08:
            _set_behavior(
                agent,
                "keeper_goal_line_turn_clear",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                raw_vx=round(vx, 2),
            )
            lane_vx, lane_vy = _map_vector_to_robot(agent, clear_dx, clear_dy)
            agent.cmd_vel(
                clamp(1.05 * lane_vx, 0.02, 0.72),
                clamp(1.05 * lane_vy + 0.18 * b_y, -0.85, 0.85),
                clamp(
                    0.6 * math.atan2(b_y, max(abs(b_x), 0.08)) * (1.0 if b_x >= 0.0 else -1.0),
                    -0.75,
                    0.75,
                ),
            )
            agent.move_head(math.inf, math.inf)
            return
        vx = clamp(1.25 * vx, 0.72, 1.0)
        vy = clamp(0.65 * vy + 0.35 * b_y - 0.25 * ball_map[1], -0.75, 0.75)
        w = clamp(0.22 * math.atan2(b_y, max(b_x, 0.04)), -0.28, 0.28)
        _set_behavior(
            agent,
            "keeper_goal_line_emergency_clear",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    if ball_dist > 0.72:
        target_x = clamp(ball_map[0] - 0.48, own_goal_x + 0.22, own_goal_x + 0.95)
        target_y = clamp(ball_map[1] * 0.72, -0.78, 0.78)
        field_target_x = own_goal_x + 1.65
        field_target_y = clamp(ball_map[1] * 0.18, -0.55, 0.55)
        _set_behavior(
            agent,
            "keeper_clear_setup",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(
            agent, target_x, target_y, face_x=field_target_x, face_y=field_target_y, stop_dist=0.13
        )
        return

    if ball_map[0] < own_goal_x + 0.55:
        field_target_x = own_goal_x + 1.45
        field_target_y = clamp(ball_map[1] * 0.25, -0.65, 0.65)
        clear_yaw_deg = math.degrees(
            math.atan2(
                field_target_y - agent.get_self_pos()[1], field_target_x - agent.get_self_pos()[0]
            )
        )
        clear_err_deg = normalize_angle_deg(clear_yaw_deg - agent.get_self_yaw())
        if b_x < 0.0 or abs(b_y) > 0.34 or abs(clear_err_deg) > 34.0:
            setup_x = clamp(ball_map[0] - 0.30, own_goal_x + 0.28, own_goal_x + 0.62)
            setup_y = clamp(ball_map[1] * 0.72, -0.72, 0.72)
            _set_behavior(
                agent,
                "keeper_goal_line_setup",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_map[0], 2),
                ball_y=round(ball_map[1], 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                yaw_err=round(clear_err_deg, 1),
            )
            _drive_to_pose(
                agent,
                setup_x,
                setup_y,
                face_x=field_target_x,
                face_y=field_target_y,
                stop_dist=0.11,
            )
            agent.move_head(math.inf, math.inf)
            return

        side = 1.0 if ball_map[1] <= 0.0 else -1.0
        target_x = field_target_x
        target_y = clamp(ball_map[1] * 0.2 + side * 0.22, -0.72, 0.72)
        yaw_rad = math.radians(agent.get_self_yaw())
        clear_dx = target_x - agent.get_self_pos()[0]
        clear_dy = target_y - agent.get_self_pos()[1]
        vx = clear_dx * math.cos(yaw_rad) + clear_dy * math.sin(yaw_rad)
        vy = -clear_dx * math.sin(yaw_rad) + clear_dy * math.cos(yaw_rad)
        vx = clamp(1.15 * vx, 0.65, 1.0)
        vy = clamp(0.55 * vy + 0.28 * b_y - 0.18 * ball_map[1], -0.45, 0.45)
        w = clamp(0.32 * math.atan2(b_y, max(b_x, 0.04)), -0.32, 0.32)
        _set_behavior(
            agent,
            "keeper_goal_line_sweep",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_map[0], 2),
            ball_y=round(ball_map[1], 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    vx = 1.0 if b_x > -0.02 else 0.55
    vy = clamp(0.95 * b_y - 0.55 * ball_map[1], -0.85, 0.85)
    w = clamp(0.65 * math.atan2(b_y, max(b_x, 0.04)), -0.65, 0.65)
    _set_behavior(
        agent,
        "keeper_fast_clear",
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_map[0], 2),
        ball_y=round(ball_map[1], 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)


def _keeper_own_goal_kick_clear(agent, tracker, ball_map):
    if not agent.get_if_ball():
        _set_behavior(agent, "own_goal_kick_keeper_find_clear")
        agent.state_machine_runners["find_ball"]()
        return

    half_len = agent.field_length / 2.0
    own_goal_x = -half_len
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    pos = agent.get_self_pos()
    yaw_deg = agent.get_self_yaw()
    yaw_rad = math.radians(yaw_deg)

    if abs(pos[0]) > half_len + 0.8 or abs(pos[1]) > agent.field_width / 2.0 + 0.8:
        target_x = own_goal_x + 0.72
        target_y = clamp(ball_y * 0.18, -0.28, 0.28)
        _set_behavior(
            agent,
            "own_goal_kick_keeper_reset_guard",
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(agent, target_x, target_y, face_x=ball_x + 1.8, face_y=0.0, stop_dist=0.12)
        return

    side = 1.0 if ball_y <= 0.0 else -1.0

    clear_target_x = clamp(ball_x + 2.45, own_goal_x + 1.75, -1.25)
    clear_target_y = clamp(ball_y * 0.04, -0.15, 0.15)
    clear_dx = clear_target_x - ball_x
    clear_dy = clear_target_y - ball_y
    clear_norm = max(math.hypot(clear_dx, clear_dy), 1e-6)
    clear_dx /= clear_norm
    clear_dy /= clear_norm

    # Goal-kick balls sit close to our goal line. To push them forward with
    # velocity-only control, the keeper must stage on the goal side of the ball.
    side_stage_y = clamp(ball_y + side * 0.58, -0.92, 0.92)
    setup_x = clamp(ball_x - 0.42, own_goal_x + 0.08, ball_x - 0.12)
    setup_y = side_stage_y
    raw_vx = clear_dx * math.cos(yaw_rad) + clear_dy * math.sin(yaw_rad)
    raw_vy = -clear_dx * math.sin(yaw_rad) + clear_dy * math.cos(yaw_rad)
    clear_yaw_deg = math.degrees(math.atan2(clear_target_y - pos[1], clear_target_x - pos[0]))
    clear_err_deg = normalize_angle_deg(clear_yaw_deg - yaw_deg)

    ready = ball_dist < 0.62 and b_x > -0.04 and abs(b_y) < 0.46 and raw_vx > 0.08
    if not ready:
        _set_behavior(
            agent,
            "own_goal_kick_keeper_safe_setup",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            clear_x=round(clear_target_x, 2),
            clear_y=round(clear_target_y, 2),
        )
        _drive_to_pose(
            agent, setup_x, setup_y, face_x=clear_target_x, face_y=clear_target_y, stop_dist=0.08
        )
        agent.move_head(math.inf, math.inf)
        return

    vx = clamp(1.24 * raw_vx, 0.68, 1.0)
    vy = clamp(0.58 * raw_vy + 0.10 * b_y, -0.38, 0.38)
    w = clamp(
        math.radians(clear_err_deg) * 0.5 + 0.10 * math.atan2(b_y, max(b_x, 0.04)), -0.58, 0.58
    )
    _set_behavior(
        agent,
        "own_goal_kick_keeper_safe_clear",
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        raw_vx=round(raw_vx, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)


def _keeper_anchor(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    half_len = agent.field_length / 2.0
    own_goal_x = -half_len + 0.15
    default_x = -half_len + 0.55

    if ball_map is None:
        _set_behavior(agent, "keeper_hold")
        _drive_to_pose(agent, default_x, 0.0, face_yaw_deg=0.0, stop_dist=0.1)
        return

    target_y = clamp(ball_map[1] * 0.7, -0.9, 0.9)
    target_x = default_x
    danger_x = -half_len + 1.6
    critical_x = -half_len + 1.0

    if ball_map[0] < danger_x:
        target_x = clamp(ball_map[0] - 0.3, own_goal_x + 0.25, -half_len + 1.2)
    if (
        ball_map[0] < critical_x
        and abs(ball_map[1]) < 1.18
        and agent.get_if_ball()
        and agent.get_ball_distance() < 0.9
    ):
        _set_behavior(agent, "keeper_clearance")
        _keeper_clear_ball(agent, tracker, ball_map)
        return

    _set_behavior(agent, "keeper_guard", target_x=round(target_x, 2), target_y=round(target_y, 2))
    _drive_to_pose(
        agent, target_x, target_y, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.12
    )


def _keep_outside_radius(ball_x, ball_y, target_x, target_y, radius, field_length, field_width):
    dx = target_x - ball_x
    dy = target_y - ball_y
    dist = math.hypot(dx, dy)
    if dist < radius:
        if dist < 1e-4:
            dx, dy, dist = -radius, 0.0, radius
        scale = radius / dist
        target_x = ball_x + dx * scale
        target_y = ball_y + dy * scale

    half_len = field_length / 2.0
    half_wid = field_width / 2.0
    return (
        clamp(target_x, -half_len + 0.25, half_len - 0.25),
        clamp(target_y, -half_wid + 0.25, half_wid - 0.25),
    )


def _opponent_setplay_defense(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        tx, ty, tyaw = _get_ready_pose(agent)
        _set_behavior(agent, "setplay_defense_reset")
        _drive_to_pose(agent, tx, ty, face_yaw_deg=tyaw, stop_dist=0.15)
        return

    if agent.role == "keeper":
        _keeper_anchor(agent, tracker)
        return
    if agent.role == "sweeper":
        _sweeper_anchor(agent, tracker)
        return

    half_len = agent.field_length / 2.0
    avoid_radius = 1.62
    ball_x, ball_y = ball_map
    side_sign = -1.0
    if str(getattr(agent, "color", agent.get_config().get("color", "red"))).lower() == "blue":
        side_sign = 1.0

    lateral = -0.85 if agent.role == "striker" else 0.85
    if abs(ball_y) > agent.field_width * 0.32:
        lateral = -0.65 if ball_y > 0 else 0.65

    target_x = ball_x + side_sign * avoid_radius
    target_y = ball_y + lateral
    if agent.role == "striker":
        # Striker takes the forward interception lane without entering the 1.5m exclusion area.
        target_x = ball_x + side_sign * (avoid_radius + 0.15)
    else:
        # Support sits slightly deeper to catch loose balls after the opponent's restart.
        target_x = ball_x + side_sign * (avoid_radius + 0.35)

    target_x = clamp(target_x, -half_len + 0.9, half_len - 0.9)

    target_x, target_y = _keep_outside_radius(
        ball_x,
        ball_y,
        target_x,
        target_y,
        avoid_radius,
        agent.field_length,
        agent.field_width,
    )
    _set_behavior(
        agent,
        "setplay_defense_lane",
        set_play=agent.gamecontroller.set_play_name,
        target_x=round(target_x, 2),
        target_y=round(target_y, 2),
    )
    _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.14)


def _own_goal_kick_play(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        if agent.role == "keeper":
            _keeper_anchor(agent, tracker)
        else:
            _set_behavior(agent, "own_goal_kick_find_ball")
            agent.state_machine_runners["find_ball"]()
        return

    half_len = agent.field_length / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])

    if agent.role == "keeper":
        if ball_x < -half_len + 0.98 and abs(ball_y) < 1.18:
            target_x = -half_len + 0.92
            target_y = clamp(ball_y * 0.18, -0.32, 0.32)
            _set_behavior(
                agent,
                "own_goal_kick_keeper_line_guard",
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
            )
            _drive_to_pose(
                agent, target_x, target_y, face_x=ball_x + 1.8, face_y=0.0, stop_dist=0.1
            )
            return
        if ball_x < -half_len + 1.05 and abs(ball_y) < 1.25:
            _keeper_own_goal_kick_clear(agent, tracker, ball_map)
            return
        target_x = -half_len + 1.08
        target_y = clamp(ball_y * 0.18, -0.28, 0.28)
        _set_behavior(
            agent,
            "own_goal_kick_keeper_hold",
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(agent, target_x, target_y, face_x=ball_x + 1.8, face_y=0.0, stop_dist=0.1)
        return

    agent_local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
    executor_id = _choose_nonkeeper_executor(tracker, ball_map)
    if ball_x < -half_len + 0.82 and abs(ball_y) < 1.15 and executor_id != agent_local_id:
        side = -1.0 if agent.role == "striker" else 1.0
        target_x = clamp(ball_x + 1.55, -half_len + 1.45, -2.35)
        target_y = clamp(side * 1.18, -1.35, 1.35)
        _set_behavior(
            agent,
            "own_goal_kick_avoid_ball",
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(agent, target_x, target_y, face_x=ball_x + 1.4, face_y=0.0, stop_dist=0.18)
        return

    candidates = []
    striker_candidate = None
    nearest_candidate = None
    for player_id, info in tracker.players.items():
        if info["penalty"] != 0 or info["role"] == "keeper":
            continue
        px, py = info["pos"]
        dist = math.hypot(ball_x - px, ball_y - py)
        candidate = (dist, player_id, info["role"])
        if nearest_candidate is None or candidate[0] < nearest_candidate[0]:
            nearest_candidate = candidate
        if info["role"] == "striker":
            striker_candidate = candidate

    if striker_candidate is not None and nearest_candidate is not None:
        striker_dist, striker_id, _ = striker_candidate
        nearest_dist, nearest_id, _ = nearest_candidate
        if striker_dist <= 1.65 or striker_dist <= nearest_dist + 0.75:
            candidates.append((striker_dist - 0.35, striker_id))
        else:
            candidates.append((nearest_dist, nearest_id))
    elif nearest_candidate is not None:
        candidates.append((nearest_candidate[0], nearest_candidate[1]))

    if candidates and min(candidates)[1] != agent_local_id:
        target_x = clamp(ball_x + 0.95, -half_len + 0.9, -0.85)
        target_y = clamp(ball_y + (0.85 if agent.role == "support" else -0.85), -1.15, 1.15)
        _set_behavior(
            agent,
            "own_goal_kick_second_ball",
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.15)
        return

    if not agent.get_if_ball():
        _set_behavior(agent, "own_goal_kick_chaser")
        _drive_to_pose(
            agent,
            clamp(ball_x + 0.78, -half_len + 0.92, -2.55),
            clamp(ball_y * 0.45, -0.72, 0.72),
            face_x=ball_x,
            face_y=ball_y,
            stop_dist=0.16,
        )
        return

    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    pos = agent.get_self_pos()
    yaw_rad = math.radians(agent.get_self_yaw())
    field_target_x = clamp(ball_x + 2.35, -half_len + 1.85, -0.7)
    field_target_y = clamp(ball_y * 0.08, -0.18, 0.18)
    if ball_x < -half_len + 0.95 and ball_dist > 1.05:
        target_x = clamp(ball_x + 0.48, -half_len + 0.62, -2.55)
        target_y = clamp(ball_y * 0.82, -0.62, 0.62)
        _set_behavior(
            agent,
            "own_goal_kick_direct_chase",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(
            agent, target_x, target_y, face_x=field_target_x, face_y=field_target_y, stop_dist=0.12
        )
        return
    if ball_x < -half_len + 0.9 and (ball_dist > 0.38 or b_x < 0.04 or abs(b_y) > 0.24):
        side = -1.0 if pos[1] <= ball_y else 1.0
        if abs(ball_y) < 0.14:
            side = -1.0 if agent.role == "striker" else 1.0
        target_x = clamp(ball_x + 0.55, -half_len + 0.72, -2.45)
        target_y = clamp(ball_y + side * 0.32, -0.72, 0.72)
        _set_behavior(
            agent,
            "own_goal_kick_safe_stage",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(
            agent, target_x, target_y, face_x=field_target_x, face_y=field_target_y, stop_dist=0.13
        )
        return

    target_dx = field_target_x - ball_x
    target_dy = field_target_y - ball_y
    target_norm = max(math.hypot(target_dx, target_dy), 1e-6)
    target_dx /= target_norm
    target_dy /= target_norm
    setup_x = clamp(ball_x + 0.55, -half_len + 0.9, -2.35)
    setup_y = clamp(ball_y - 0.35 * target_dy, -0.72, 0.72)
    target_x = field_target_x
    target_y = field_target_y
    clear_dx = target_x - ball_x
    clear_dy = target_y - ball_y
    clear_norm = max(math.hypot(clear_dx, clear_dy), 1e-6)
    clear_dx /= clear_norm
    clear_dy /= clear_norm
    raw_vx = clear_dx * math.cos(yaw_rad) + clear_dy * math.sin(yaw_rad)
    raw_vy = -clear_dx * math.sin(yaw_rad) + clear_dy * math.cos(yaw_rad)

    if ball_dist > 0.55 or raw_vx < 0.38 or b_x < 0.04 or abs(b_y) > 0.20:
        _set_behavior(
            agent,
            "own_goal_kick_setup",
            ball_dist=round(ball_dist, 3),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.12)
        return

    clear_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
    clear_err_deg = normalize_angle_deg(clear_yaw_deg - agent.get_self_yaw())
    vx = clamp(0.95 * raw_vx, 0.42, 0.72)
    vy = clamp(0.7 * raw_vy + 0.12 * b_y, -0.38, 0.38)
    w = clamp(
        math.radians(clear_err_deg) * 0.5 + 0.18 * math.atan2(b_y, max(b_x, 0.04)), -0.55, 0.55
    )
    _set_behavior(
        agent,
        "own_goal_kick_clear",
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)


def _choose_nonkeeper_executor(tracker, ball_map):
    candidates = []
    for local_id, info in tracker.players.items():
        if info["penalty"] != 0 or info["role"] == "keeper":
            continue
        px, py = info["pos"]
        dist = math.hypot(ball_map[0] - px, ball_map[1] - py)
        role_bias = -0.08 if info["role"] == "support" else 0.0
        candidates.append((dist + role_bias, local_id))
    if not candidates:
        return None
    return min(candidates)[1]


def _own_sideline_setplay(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        if agent.role == "keeper":
            _keeper_anchor(agent, tracker)
        else:
            _set_behavior(agent, "own_sideline_find_ball")
            agent.state_machine_runners["find_ball"]()
        return

    if agent.role == "keeper":
        _keeper_anchor(agent, tracker)
        return

    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    goal_x = half_len
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    side = 1.0 if ball_y >= 0.0 else -1.0
    local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
    executor_id = _choose_nonkeeper_executor(tracker, ball_map)

    if executor_id is not None and executor_id != local_id:
        receive_x = clamp(
            ball_x + (1.55 if ball_x < 1.6 else 0.85), -half_len + 1.05, goal_x - 0.35
        )
        receive_y = clamp(-0.35 * side, -0.55, 0.55)
        if ball_x > goal_x - 1.35:
            receive_x = goal_x - 0.55
            receive_y = clamp(side * 0.24, -0.34, 0.34)
        _set_behavior(
            agent,
            "own_sideline_second_ball",
            set_play=agent.gamecontroller.set_play_name,
            target_x=round(receive_x, 2),
            target_y=round(receive_y, 2),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
        )
        _drive_to_pose(agent, receive_x, receive_y, face_x=ball_x, face_y=ball_y, stop_dist=0.14)
        return

    if not agent.get_if_ball():
        setup_x = clamp(ball_x - 0.35, -half_len + 0.65, goal_x - 0.35)
        setup_y = clamp(ball_y, -half_wid + 0.12, half_wid - 0.12)
        _set_behavior(
            agent,
            "own_sideline_executor_approach",
            set_play=agent.gamecontroller.set_play_name,
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=ball_x, face_y=ball_y, stop_dist=0.12)
        return

    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    pos = agent.get_self_pos()
    yaw_deg = agent.get_self_yaw()
    yaw_rad = math.radians(yaw_deg)

    if ball_x > goal_x - 1.45 and abs(ball_y) < half_wid - 0.12 and ball_dist < 1.5:
        if _finish_attack(agent, tracker, ball_map):
            _set_behavior(
                agent,
                "own_sideline_goalmouth_finish",
                set_play=agent.gamecontroller.set_play_name,
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_x, 2),
                ball_y=round(ball_y, 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
            )
            return

    if ball_x > goal_x - 1.35 and abs(ball_y) > 1.35 and ball_dist < 1.45:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = goal_x - (0.28 if abs(ball_y) > 2.25 else 0.12)
        target_y = side * (0.52 if abs(ball_y) > 2.25 else 0.28)
        cut_dx = target_x - ball_x
        cut_dy = target_y - ball_y
        cut_norm = max(math.hypot(cut_dx, cut_dy), 1e-6)
        cut_dx /= cut_norm
        cut_dy /= cut_norm
        raw_vx = cut_dx * math.cos(yaw_rad) + cut_dy * math.sin(yaw_rad)
        raw_vy = -cut_dx * math.sin(yaw_rad) + cut_dy * math.cos(yaw_rad)
        cut_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        cut_err_deg = normalize_angle_deg(cut_yaw_deg - yaw_deg)
        vx = clamp(0.55 * raw_vx + 0.04 * b_x, -0.04, 0.42)
        if ball_x > goal_x - 0.45:
            vx = min(vx, 0.14)
        vy = clamp(1.75 * raw_vy, -1.0, 1.0)
        w = clamp(
            math.radians(cut_err_deg) * 0.92 + 0.08 * math.atan2(b_y, max(b_x, 0.04)), -0.95, 0.95
        )
        _set_behavior(
            agent,
            "own_sideline_deep_cut_in",
            set_play=agent.gamecontroller.set_play_name,
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    if ball_x > goal_x - 0.65 and abs(ball_y) > half_wid - 0.35 and ball_dist < 1.9:
        target_x = goal_x - 0.08
        target_y = side * 0.42
        drive_dx = target_x - ball_x
        drive_dy = target_y - ball_y
        drive_norm = max(math.hypot(drive_dx, drive_dy), 1e-6)
        drive_dx /= drive_norm
        drive_dy /= drive_norm
        raw_vx = drive_dx * math.cos(yaw_rad) + drive_dy * math.sin(yaw_rad)
        raw_vy = -drive_dx * math.sin(yaw_rad) + drive_dy * math.cos(yaw_rad)
        drive_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        drive_err_deg = normalize_angle_deg(drive_yaw_deg - yaw_deg)
        if ball_dist > 0.92 and b_x < -0.08:
            setup_x = clamp(ball_x - 0.25 * drive_dx, goal_x - 0.85, goal_x - 0.18)
            setup_y = clamp(ball_y - 0.25 * drive_dy, -half_wid + 0.08, half_wid - 0.08)
            _set_behavior(
                agent,
                "own_sideline_front_corner_fast_setup",
                set_play=agent.gamecontroller.set_play_name,
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_x, 2),
                ball_y=round(ball_y, 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
            )
            _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.1)
            agent.move_head(math.inf, math.inf)
            return

        vx = clamp(0.68 * raw_vx, 0.04, 0.38)
        vy = clamp(1.75 * raw_vy, -1.0, 1.0)
        if b_x < -0.18:
            vx = min(vx, 0.16)
        w = clamp(
            math.radians(drive_err_deg) * 0.88 + 0.08 * math.atan2(b_y, max(b_x, 0.04)), -0.86, 0.86
        )
        _set_behavior(
            agent,
            "own_sideline_front_corner_sweep",
            set_play=agent.gamecontroller.set_play_name,
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    if -0.65 < ball_x < goal_x - 1.0 and abs(ball_y) > half_wid - 0.55 and ball_dist < 1.35:
        target_x = clamp(ball_x + 1.05, 0.45, goal_x - 1.05)
        target_y = clamp(side * 0.62, -0.78, 0.78)
        drive_dx = target_x - ball_x
        drive_dy = target_y - ball_y
        drive_norm = max(math.hypot(drive_dx, drive_dy), 1e-6)
        drive_dx /= drive_norm
        drive_dy /= drive_norm
        raw_vx = drive_dx * math.cos(yaw_rad) + drive_dy * math.sin(yaw_rad)
        raw_vy = -drive_dx * math.sin(yaw_rad) + drive_dy * math.cos(yaw_rad)
        drive_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        drive_err_deg = normalize_angle_deg(drive_yaw_deg - yaw_deg)
        if ball_dist < 1.05 and b_x > -0.85 and abs(b_y) < 1.05:
            vx = clamp(0.82 * raw_vx + 0.08 * b_x, 0.16, 0.72)
            vy = clamp(1.35 * raw_vy + 0.18 * b_y, -1.0, 1.0)
            if raw_vx < 0.02:
                vx = min(vx, 0.24)
            w = clamp(
                math.radians(drive_err_deg) * 0.78 + 0.12 * math.atan2(b_y, max(b_x, 0.04)),
                -0.82,
                0.82,
            )
            _set_behavior(
                agent,
                "own_sideline_midfield_force_cut",
                set_play=agent.gamecontroller.set_play_name,
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_x, 2),
                ball_y=round(ball_y, 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                raw_vx=round(raw_vx, 2),
                raw_vy=round(raw_vy, 2),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return
        setup_x = clamp(ball_x - 0.18, -half_len + 0.65, goal_x - 1.15)
        setup_y = clamp(ball_y - side * 0.58, -half_wid + 1.0, half_wid - 1.0)
        ready = ball_dist < 0.72 and b_x > -0.05 and abs(b_y) < 0.62
        if not ready:
            _set_behavior(
                agent,
                "own_sideline_midfield_inside_setup",
                set_play=agent.gamecontroller.set_play_name,
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_x, 2),
                ball_y=round(ball_y, 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                drive_x=round(target_x, 2),
                drive_y=round(target_y, 2),
            )
            _drive_to_pose(
                agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.11
            )
            agent.move_head(math.inf, math.inf)
            return

        vx = clamp(0.92 * raw_vx + 0.12 * b_x, 0.22, 0.82)
        vy = clamp(1.48 * raw_vy + 0.20 * b_y, -1.0, 1.0)
        w = clamp(
            math.radians(drive_err_deg) * 0.72 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.78, 0.78
        )
        _set_behavior(
            agent,
            "own_sideline_midfield_cut_in",
            set_play=agent.gamecontroller.set_play_name,
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    target_x = clamp(ball_x + (1.95 if ball_x < 1.6 else 1.15), -half_len + 1.15, goal_x - 0.42)
    target_y = clamp(-0.18 * side if ball_x < goal_x - 1.25 else 0.0, -0.42, 0.42)
    if ball_x > goal_x - 1.25:
        target_x = goal_x + 0.25
        target_y = 0.0

    drive_dx = target_x - ball_x
    drive_dy = target_y - ball_y
    drive_norm = max(math.hypot(drive_dx, drive_dy), 1e-6)
    drive_dx /= drive_norm
    drive_dy /= drive_norm
    setup_x = clamp(ball_x - 0.42 * drive_dx, -half_len + 0.55, goal_x - 0.25)
    setup_y = clamp(ball_y - 0.42 * drive_dy, -half_wid + 0.08, half_wid - 0.08)
    raw_vx = drive_dx * math.cos(yaw_rad) + drive_dy * math.sin(yaw_rad)
    raw_vy = -drive_dx * math.sin(yaw_rad) + drive_dy * math.cos(yaw_rad)
    drive_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
    drive_err_deg = normalize_angle_deg(drive_yaw_deg - yaw_deg)

    ready = ball_dist < 0.74 and b_x > 0.0 and abs(b_y) < 0.62 and raw_vx > 0.02
    if not ready:
        _set_behavior(
            agent,
            "own_sideline_restart_setup",
            set_play=agent.gamecontroller.set_play_name,
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            drive_x=round(target_x, 2),
            drive_y=round(target_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return

    vx = clamp(1.12 * raw_vx, 0.34, 0.9)
    vy = clamp(1.05 * raw_vy + 0.12 * b_y, -0.82, 0.82)
    w = clamp(
        math.radians(drive_err_deg) * 0.55 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.58, 0.58
    )
    _set_behavior(
        agent,
        "own_sideline_restart_drive",
        set_play=agent.gamecontroller.set_play_name,
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        raw_vx=round(raw_vx, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)


def _own_defensive_sideline_setplay(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        if agent.role == "keeper":
            _keeper_anchor(agent, tracker)
        else:
            _set_behavior(agent, "own_def_sideline_find_ball")
            agent.state_machine_runners["find_ball"]()
        return

    if agent.role == "keeper":
        _keeper_anchor(agent, tracker)
        return

    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    side = 1.0 if ball_y >= 0.0 else -1.0
    local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
    executor_id = _choose_nonkeeper_executor(tracker, ball_map)

    if executor_id is not None and executor_id != local_id:
        receive_x = clamp(ball_x + 1.35, -half_len + 1.0, 0.85)
        receive_y = clamp(-0.35 * side, -0.75, 0.75)
        _set_behavior(
            agent,
            "own_def_sideline_second_ball",
            set_play=agent.gamecontroller.set_play_name,
            target_x=round(receive_x, 2),
            target_y=round(receive_y, 2),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
        )
        _drive_to_pose(agent, receive_x, receive_y, face_x=ball_x, face_y=ball_y, stop_dist=0.15)
        return

    if ball_x < -half_len + 0.85:
        hold_x = clamp(ball_x + 1.15, -half_len + 0.95, -1.25)
        hold_y = clamp(-0.35 * side, -0.75, 0.75)
        _set_behavior(
            agent,
            "own_def_sideline_screen_hold",
            set_play=agent.gamecontroller.set_play_name,
            target_x=round(hold_x, 2),
            target_y=round(hold_y, 2),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
        )
        _drive_to_pose(agent, hold_x, hold_y, face_x=ball_x + 1.4, face_y=0.0, stop_dist=0.15)
        return

    clear_target_x = clamp(ball_x + 2.45, -half_len + 1.35, 1.2)
    clear_target_y = clamp(ball_y * 0.08, -0.22, 0.22)
    if not agent.get_if_ball():
        setup_x = clamp(ball_x + 0.42, -half_len + 0.55, 0.35)
        setup_y = clamp(ball_y - side * 0.68, -half_wid + 0.28, half_wid - 0.28)
        _set_behavior(
            agent,
            "own_def_sideline_executor_approach",
            set_play=agent.gamecontroller.set_play_name,
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
        )
        _drive_to_pose(
            agent, setup_x, setup_y, face_x=clear_target_x, face_y=clear_target_y, stop_dist=0.13
        )
        return

    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    pos = agent.get_self_pos()
    yaw_deg = agent.get_self_yaw()
    yaw_rad = math.radians(yaw_deg)
    drive_dx = clear_target_x - ball_x
    drive_dy = clear_target_y - ball_y
    drive_norm = max(math.hypot(drive_dx, drive_dy), 1e-6)
    drive_dx /= drive_norm
    drive_dy /= drive_norm
    raw_vx = drive_dx * math.cos(yaw_rad) + drive_dy * math.sin(yaw_rad)
    raw_vy = -drive_dx * math.sin(yaw_rad) + drive_dy * math.cos(yaw_rad)
    setup_x = clamp(ball_x - 0.42 * drive_dx, -half_len + 0.55, 0.35)
    setup_y = clamp(ball_y - 0.42 * drive_dy, -half_wid + 0.24, half_wid - 0.24)
    drive_yaw_deg = math.degrees(math.atan2(clear_target_y - pos[1], clear_target_x - pos[0]))
    drive_err_deg = normalize_angle_deg(drive_yaw_deg - yaw_deg)

    ready = ball_dist < 0.82 and b_x > -0.18 and abs(b_y) < 0.72 and raw_vx > -0.08
    if not ready:
        _set_behavior(
            agent,
            "own_def_sideline_restart_setup",
            set_play=agent.gamecontroller.set_play_name,
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            clear_x=round(clear_target_x, 2),
            clear_y=round(clear_target_y, 2),
        )
        _drive_to_pose(
            agent, setup_x, setup_y, face_x=clear_target_x, face_y=clear_target_y, stop_dist=0.11
        )
        agent.move_head(math.inf, math.inf)
        return

    vx = clamp(1.12 * raw_vx, 0.34, 0.9)
    vy = clamp(1.0 * raw_vy + 0.14 * b_y, -0.82, 0.82)
    w = clamp(
        math.radians(drive_err_deg) * 0.55 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.58, 0.58
    )
    _set_behavior(
        agent,
        "own_def_sideline_restart_drive",
        set_play=agent.gamecontroller.set_play_name,
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        raw_vx=round(raw_vx, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)


def _cup_nonkeeper_handler_id(tracker, ball_map, prefer_support=False):
    """Unified 3v3 executor selection: one ball carrier, one forward outlet."""
    candidates = []
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    half_len = tracker.field_length / 2.0
    deep_defense = ball_x < -half_len + 1.45
    own_kickoff_start = (
        tracker.current_set_play == "SET_PLAY_KICK_OFF"
        and tracker.current_can_kick
        and abs(ball_x) < 0.75
        and abs(ball_y) < 0.65
    )
    for local_id, info in tracker.players.items():
        if info["penalty"] != 0 or info["role"] == "keeper":
            continue
        px, py = info["pos"]
        dist = math.hypot(ball_x - px, ball_y - py)
        lateral = abs(ball_y - py)
        role = info["role"]
        score = dist
        soft_out = (
            abs(px) > tracker.field_length / 2.0 - 0.05
            or abs(py) > tracker.field_width / 2.0 - 0.05
        )
        hard_out = (
            abs(px) > tracker.field_length / 2.0 + 0.16
            or abs(py) > tracker.field_width / 2.0 + 0.16
        )
        if soft_out:
            score += 1.4
        if hard_out:
            score += 4.5
        if own_kickoff_start:
            # Kickoff is a scripted center-lane VLA plan: striker makes first
            # contact only if already aligned; otherwise the closest in-line
            # nonkeeper executes while the other robot becomes an outlet.
            behind_center = px < ball_x - 0.12 and abs(py - ball_y) < 1.2
            if behind_center:
                score -= 0.95
            if role == "striker":
                score -= 0.35
            elif role == "support":
                score -= 0.20
            else:
                score += 0.20
        if abs(ball_y) > tracker.field_width / 2.0 - 0.62:
            side = 1.0 if ball_y > 0.0 else -1.0
            inside_gap = side * (ball_y - py)
            if inside_gap > 0.35:
                score -= 0.45
            else:
                score += 0.85
        if prefer_support and role == "support" and lateral <= 1.05:
            score -= 0.45
        if deep_defense:
            # In our box, the field players must own the clearance so the keeper
            # can stay on the line instead of joining repeated contact loops.
            if role == "support" and lateral <= 1.25:
                score -= 0.48
            if px > ball_x + 0.12:
                score -= 0.18
            if abs(py - ball_y) < 0.85:
                score -= 0.14
        elif ball_x > 1.2:
            if role == "striker":
                score -= 0.28
            if px > ball_x + 0.2:
                score += 0.55
        else:
            if role == "support":
                score -= 0.18
        if lateral > 1.15:
            score += 1.25
        if lateral > 1.65:
            score += 2.2
        candidates.append((score, local_id))
    if not candidates:
        return None
    return min(candidates)[1]


def _embodied_scene(agent, tracker, ball_map):
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    score_margin = int(agent.gamecontroller.score) - int(agent.gamecontroller.opponent_score)
    secs_remaining = agent.gamecontroller.secs_remaining
    try:
        secs_remaining = int(secs_remaining)
    except (TypeError, ValueError):
        secs_remaining = 600
    phase = _field_phase(agent, ball_map)
    sideline = abs(ball_y) > half_wid - 0.58
    corner = abs(ball_y) > half_wid - 0.58 and (ball_x > half_len - 1.2 or ball_x < -half_len + 1.2)
    own_danger = ball_x < -half_len + 1.55 and abs(ball_y) < 1.35
    own_wide_danger = ball_x < -0.9 and abs(ball_y) > 1.65
    front_goal = ball_x > half_len - 1.45 and abs(ball_y) < 1.2
    protect_lead = score_margin >= 2 and secs_remaining <= 180
    protect_lead = protect_lead or (score_margin >= 1 and secs_remaining <= 90)
    return {
        "phase": phase,
        "score_margin": score_margin,
        "secs_remaining": secs_remaining,
        "protect_lead": protect_lead,
        "sideline": sideline,
        "corner": corner,
        "own_danger": own_danger,
        "own_wide_danger": own_wide_danger,
        "front_goal": front_goal,
        "ball_x": ball_x,
        "ball_y": ball_y,
    }


def _embodied_intent(agent, tracker, ball_map, set_play_name=None):
    scene = _embodied_scene(agent, tracker, ball_map)
    gc = agent.gamecontroller
    own_kickoff = (
        gc.set_play_name == "SET_PLAY_KICK_OFF"
        and bool(gc.can_kick)
        and gc.kicking_team == gc.team
        and -0.45 < float(ball_map[0]) < 0.85
        and abs(float(ball_map[1])) < 0.65
    )
    if own_kickoff:
        return "kickoff_playbook", scene
    if gc.set_play_name not in ("SET_PLAY_NONE", None) and not gc.can_kick:
        return "defend_restart", scene
    if set_play_name == "SET_PLAY_GOAL_KICK" or scene["own_danger"] or scene.get("own_wide_danger"):
        return "clear_own_box", scene
    if scene.get("own_wide_danger"):
        return "clear_own_box", scene
    if (
        -0.25 < float(ball_map[0]) < 1.25
        and 1.15 < abs(float(ball_map[1])) < agent.field_width / 2.0 - 0.45
    ):
        return "recover_mid_wide", scene
    if scene["sideline"]:
        return "escape_sideline", scene
    if scene.get("protect_lead") and ball_map[0] < agent.field_length / 2.0 - 0.75:
        return "protect_lead", scene
    if scene["front_goal"]:
        return "finish_goal", scene
    if scene["phase"] in ("advance", "wide_finish", "finish"):
        return "advance_finish", scene
    if scene["phase"] in ("defend", "defend_box"):
        return "progress_out", scene
    return "progress_midfield", scene


def _set_embodied_intent(agent, tracker, intent, scene):
    now = time.time()
    if (
        intent != getattr(agent, "embodied_last_intent", None)
        or now - getattr(agent, "embodied_last_intent_ts", 0.0) > 1.0
    ):
        tracker.recorder.log(
            "embodied_intent",
            robot=int(getattr(agent, "local_id", agent.get_config().get("id", 0))),
            role=agent.role,
            intent=intent,
            phase=scene.get("phase"),
            score_margin=scene.get("score_margin"),
            ball_x=round(scene.get("ball_x", 0.0), 3),
            ball_y=round(scene.get("ball_y", 0.0), 3),
            sideline=bool(scene.get("sideline")),
            own_danger=bool(scene.get("own_danger")),
        )
        agent.embodied_last_intent = intent
        agent.embodied_last_intent_ts = now


def _sim_opponents(agent):
    vision = getattr(agent, "_vision", None)
    if vision is None:
        return []
    return list(getattr(vision, "opponents_map", []) or [])


def _segment_distance_and_progress(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-8:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t_clamped = clamp(t, 0.0, 1.0)
    cx = ax + t_clamped * dx
    cy = ay + t_clamped * dy
    return math.hypot(px - cx, py - cy), t


def _lane_congestion(agent, ball_map, target_x, target_y, corridor=0.72):
    """Scene-graph affordance cost for a planned ball corridor."""
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    pressure = 0.0
    for opponent in _sim_opponents(agent):
        ox = float(opponent.get("x", 0.0))
        oy = float(opponent.get("y", 0.0))
        dist_to_lane, progress = _segment_distance_and_progress(
            ox, oy, ball_x, ball_y, target_x, target_y
        )
        if -0.12 <= progress <= 1.12 and dist_to_lane < corridor:
            pressure += (
                (corridor - dist_to_lane) / corridor * (1.0 + 0.35 * clamp(progress, 0.0, 1.0))
            )
        if ox > ball_x - 0.15 and abs(oy - ball_y) < 0.85 and abs(ox - ball_x) < 1.2:
            pressure += 0.35
    return pressure


def _set_embodied_plan(agent, tracker, plan_name, target_x, target_y, cost, congestion):
    now = time.time()
    if (
        plan_name != getattr(agent, "embodied_last_plan", None)
        or now - getattr(agent, "embodied_last_plan_ts", 0.0) > 1.0
    ):
        tracker.recorder.log(
            "embodied_plan",
            robot=int(getattr(agent, "local_id", agent.get_config().get("id", 0))),
            role=agent.role,
            plan=plan_name,
            target_x=round(target_x, 3),
            target_y=round(target_y, 3),
            cost=round(cost, 3),
            congestion=round(congestion, 3),
        )
        agent.embodied_last_plan = plan_name
        agent.embodied_last_plan_ts = now


def _score_cup_plan(agent, ball_map, target_x, target_y, plan_name, scene):
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    congestion = _lane_congestion(agent, ball_map, target_x, target_y)
    progress = target_x - ball_x
    lateral_shift = abs(target_y - ball_y)
    central_finish_bonus = 0.0
    if ball_x > agent.field_length / 2.0 - 1.5 and abs(target_y) < 0.22:
        central_finish_bonus = 0.36
    cost = (
        1.12 * congestion
        + 0.18 * abs(target_y)
        + 0.10 * lateral_shift
        - 0.24 * progress
        - central_finish_bonus
    )
    if scene.get("score_margin", 0) < 0:
        cost -= 0.10 * max(0.0, target_x)
    if (
        plan_name == getattr(agent, "embodied_last_plan", None)
        and time.time() - getattr(agent, "embodied_last_plan_ts", 0.0) < 4.0
    ):
        cost -= 0.26
    return cost, congestion


def _select_cup_playbook_plan(agent, tracker, ball_map, intent, scene, set_play_name=None):
    half_len = agent.field_length / 2.0
    goal_x = half_len
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])

    fixed = None
    if intent == "clear_own_box":
        if scene.get("own_wide_danger"):
            side = 1.0 if ball_y > 0.0 else -1.0
            fixed = (
                -0.45,
                clamp(-0.22 * side, -0.38, 0.38),
                "cup_vla_clear_own_wide",
                0.34,
                1.18,
                0.22,
                1.0,
                1.0,
            )
        else:
            fixed = (
                1.08,
                clamp(ball_y * 0.12, -0.34, 0.34),
                "cup_vla_clear_own_box",
                0.42,
                0.86,
                0.42,
                1.0,
                0.82,
            )
    elif intent == "progress_out":
        fixed = (
            0.95,
            clamp(ball_y * 0.18, -0.45, 0.45),
            "cup_vla_progress_out",
            0.46,
            0.82,
            0.34,
            0.96,
            0.78,
        )
    if fixed is not None:
        congestion = _lane_congestion(agent, ball_map, fixed[0], fixed[1])
        _set_embodied_plan(agent, tracker, fixed[2], fixed[0], fixed[1], -0.01, congestion)
        return fixed

    candidates = []
    if intent == "finish_goal" or ball_x > goal_x - 1.45:
        candidates.extend(
            [
                (goal_x + 0.62, 0.0, "cup_vla_finish_center_crash", 0.32, 1.04, 0.50, 1.0, 0.50),
                (
                    goal_x + 0.28,
                    0.34,
                    "cup_vla_finish_left_halfspace",
                    0.36,
                    0.98,
                    0.34,
                    0.92,
                    0.78,
                ),
                (
                    goal_x + 0.28,
                    -0.34,
                    "cup_vla_finish_right_halfspace",
                    0.36,
                    0.98,
                    0.34,
                    0.92,
                    0.78,
                ),
            ]
        )
    elif intent == "advance_finish":
        side = 1.0 if ball_y >= 0.0 else -1.0
        candidates.extend(
            [
                (goal_x + 0.35, 0.0, "cup_vla_advance_center_lane", 0.38, 0.96, 0.42, 1.0, 0.56),
                (
                    goal_x - 0.12,
                    side * 0.42,
                    "cup_vla_advance_same_halfspace",
                    0.42,
                    0.92,
                    0.20,
                    0.86,
                    0.95,
                ),
                (
                    goal_x - 0.18,
                    -side * 0.38,
                    "cup_vla_advance_switch_halfspace",
                    0.44,
                    0.92,
                    0.20,
                    0.86,
                    0.95,
                ),
            ]
        )
    else:
        candidates.extend(
            [
                (3.15, 0.0, "cup_vla_midfield_center_lane", 0.46, 0.88, 0.38, 1.0, 0.68),
                (2.85, 0.54, "cup_vla_midfield_left_lane", 0.48, 0.88, 0.32, 0.96, 0.86),
                (2.85, -0.54, "cup_vla_midfield_right_lane", 0.48, 0.88, 0.32, 0.96, 0.86),
            ]
        )

    scored = []
    for candidate in candidates:
        target_x, target_y, plan_name = candidate[0], candidate[1], candidate[2]
        cost, congestion = _score_cup_plan(agent, ball_map, target_x, target_y, plan_name, scene)
        scored.append((cost, congestion, candidate))
    scored.sort(key=lambda item: item[0])
    cost, congestion, chosen = scored[0]
    _set_embodied_plan(agent, tracker, chosen[2], chosen[0], chosen[1], cost, congestion)
    return chosen


def _cup_push_plan_for_intent(agent, tracker, ball_map, intent, scene, set_play_name=None):
    if intent == "kickoff_playbook":
        ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
        age = time.time() - getattr(tracker, "own_kickoff_started_ts", 0.0)
        if age < 7.0:
            target_x = 2.35
            target_y = clamp(ball_y * 0.08, -0.18, 0.18)
            plan = (
                target_x,
                target_y,
                "cup_vla_kickoff_center_breakout",
                0.36,
                1.05,
                0.48,
                1.0,
                0.42,
            )
        else:
            target_x = 3.25
            target_y = clamp(ball_y * 0.12, -0.30, 0.30)
            plan = (target_x, target_y, "cup_vla_kickoff_second_phase", 0.42, 0.95, 0.38, 1.0, 0.55)
        congestion = _lane_congestion(agent, ball_map, target_x, target_y)
        _set_embodied_plan(agent, tracker, plan[2], target_x, target_y, -0.18, congestion)
        return plan
    if intent == "protect_lead":
        ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
        side = 1.0 if ball_y >= 0.0 else -1.0
        target_x = clamp(ball_x + 1.45, 0.75, agent.field_length / 2.0 - 1.15)
        target_y = clamp(side * 1.05, -1.15, 1.15)
        plan = (target_x, target_y, "cup_vla_protect_lead_hold_lane", 0.44, 0.9, 0.22, 0.76, 0.72)
        congestion = _lane_congestion(agent, ball_map, target_x, target_y)
        _set_embodied_plan(agent, tracker, plan[2], target_x, target_y, -0.08, congestion)
        return plan
    if intent == "recover_mid_wide":
        ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = clamp(ball_x + 2.35, 2.15, 3.35)
        target_y = clamp(side * 0.36, -0.46, 0.46)
        plan = (target_x, target_y, "cup_vln_mid_wide_breakout", 0.36, 1.18, 0.30, 1.0, 1.0)
        congestion = _lane_congestion(agent, ball_map, target_x, target_y)
        _set_embodied_plan(agent, tracker, plan[2], target_x, target_y, -0.12, congestion)
        return plan
    if intent == "escape_sideline":
        half_len = agent.field_length / 2.0
        ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = (
            half_len - 0.2 if ball_x > half_len - 1.25 else (2.85 if ball_x > -0.45 else 0.95)
        )
        target_y = clamp(side * 0.34, -0.46, 0.46)
        plan = (target_x, target_y, "cup_vln_sideline_infield", 0.52, 0.92, 0.18, 0.82, 0.95)
        congestion = _lane_congestion(agent, ball_map, target_x, target_y)
        _set_embodied_plan(agent, tracker, plan[2], target_x, target_y, -0.01, congestion)
        return plan
    return _select_cup_playbook_plan(agent, tracker, ball_map, intent, scene, set_play_name)


def _cup_push_plan(agent, ball_map, set_play_name=None):
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    goal_x = half_len
    own_goal_x = -half_len
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    abs_y = abs(ball_y)

    if abs_y > half_wid - 0.72 and ball_x < goal_x - 0.7:
        side = 1.0 if ball_y > 0.0 else -1.0
        target_x = 3.05 if ball_x > -0.45 else 1.35
        return (
            target_x,
            clamp(side * 0.52, -0.58, 0.58),
            "cup_sideline_escape",
            0.28,
            1.08,
            0.5,
            1.0,
            0.95,
        )

    if set_play_name == "SET_PLAY_GOAL_KICK" or ball_x < own_goal_x + 1.18:
        return (
            1.08,
            clamp(ball_y * 0.12, -0.34, 0.34),
            "cup_deep_escape",
            0.42,
            0.86,
            0.42,
            1.0,
            0.82,
        )

    if ball_x < -0.75:
        return (
            0.95,
            clamp(ball_y * 0.18, -0.45, 0.45),
            "cup_defense_to_midfield",
            0.44,
            0.82,
            0.34,
            0.96,
            0.78,
        )

    if ball_x < 1.9:
        return (
            3.15,
            clamp(ball_y * 0.12, -0.36, 0.36),
            "cup_midfield_vertical",
            0.46,
            0.86,
            0.38,
            1.0,
            0.72,
        )

    if ball_x > goal_x - 1.25:
        return (
            goal_x + 0.55,
            clamp(ball_y * 0.05, -0.12, 0.12),
            "cup_goal_crash",
            0.32,
            1.02,
            0.48,
            1.0,
            0.52,
        )

    if abs_y > 1.1:
        side = 1.0 if ball_y > 0.0 else -1.0
        return (
            goal_x - 0.18,
            clamp(side * 0.42, -0.52, 0.52),
            "cup_wide_cut_to_goal",
            0.36,
            0.9,
            0.18,
            0.78,
            1.0,
        )

    return (
        goal_x + 0.35,
        0.0,
        "cup_central_finish",
        0.36,
        0.95,
        0.46,
        1.0,
        0.5,
    )


def _cup_drive_ball(agent, tracker, ball_map, set_play_name=None):
    intent, scene = _embodied_intent(agent, tracker, ball_map, set_play_name)
    _set_embodied_intent(agent, tracker, intent, scene)

    if intent in ("kickoff_playbook", "recover_mid_wide"):
        (
            target_x,
            target_y,
            behavior,
            setup_back,
            ready_dist,
            min_vx,
            max_vx,
            vy_limit,
        ) = _cup_push_plan_for_intent(agent, tracker, ball_map, intent, scene, set_play_name)
        _stage_push_to_target(
            agent,
            tracker,
            ball_map,
            target_x,
            target_y,
            behavior,
            setup_back=setup_back,
            ready_dist=ready_dist,
            min_vx=min_vx,
            max_vx=max_vx,
            vy_limit=vy_limit,
        )
        return

    if intent == "protect_lead" and _cup_protect_lead_hold(agent, tracker, ball_map, scene):
        return

    if (
        intent == "clear_own_box"
        and scene.get("own_wide_danger")
        and _cup_own_wide_open_clear(agent, tracker, ball_map)
    ):
        return

    if _cup_high_wide_cutback_finish(agent, tracker, ball_map):
        return

    if intent == "escape_sideline":
        _cup_sideline_escape_direct(agent, tracker, ball_map)
        return

    if agent.get_if_ball() and _finish_attack(agent, tracker, ball_map):
        _set_behavior(agent, "cup_finish_attack", ball_dist=round(agent.get_ball_distance(), 3))
        return

    (
        target_x,
        target_y,
        behavior,
        setup_back,
        ready_dist,
        min_vx,
        max_vx,
        vy_limit,
    ) = _cup_push_plan_for_intent(agent, tracker, ball_map, intent, scene, set_play_name)

    _stage_push_to_target(
        agent,
        tracker,
        ball_map,
        target_x,
        target_y,
        behavior,
        setup_back=setup_back,
        ready_dist=ready_dist,
        min_vx=min_vx,
        max_vx=max_vx,
        vy_limit=vy_limit,
    )


def _cup_protect_lead_hold(agent, tracker, ball_map, scene):
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    if ball_x > half_len - 0.65 and abs(ball_y) < 1.05:
        return False
    side = 1.0 if ball_y >= 0.0 else -1.0
    if ball_x < -0.85:
        target_x = 1.15
        target_y = 0.0
    elif abs(ball_y) > half_wid - 0.62:
        target_x = clamp(ball_x + 0.75, 0.45, half_len - 1.15)
        target_y = clamp(side * 0.78, -0.9, 0.9)
    else:
        target_x = clamp(ball_x + 1.10, 1.05, half_len - 1.05)
        target_y = clamp(side * 1.05, -1.15, 1.15)

    drive_dx = target_x - ball_x
    drive_dy = target_y - ball_y
    drive_norm = max(math.hypot(drive_dx, drive_dy), 1e-6)
    drive_dx /= drive_norm
    drive_dy /= drive_norm

    if not agent.get_if_ball():
        setup_x = clamp(ball_x - 0.42 * drive_dx, -half_len + 0.55, half_len - 0.75)
        setup_y = clamp(ball_y - 0.42 * drive_dy, -half_wid + 0.28, half_wid - 0.28)
        _set_behavior(
            agent,
            "cup_vla_protect_lead_hold_setup",
            score_margin=scene.get("score_margin"),
            secs_remaining=scene.get("secs_remaining"),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            hold_x=round(target_x, 2),
            hold_y=round(target_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.12)
        agent.move_head(math.inf, math.inf)
        return True

    pos = agent.get_self_pos()
    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    raw_vx, raw_vy = _map_vector_to_robot(agent, drive_dx, drive_dy)
    rel_x = pos[0] - ball_x
    rel_y = pos[1] - ball_y
    behind = rel_x * drive_dx + rel_y * drive_dy
    lateral = -rel_x * drive_dy + rel_y * drive_dx
    ready = (
        ball_dist < 1.28
        and b_x > -0.36
        and raw_vx > -0.22
        and behind < 0.42
        and abs(lateral) < 1.05
    )
    if not ready:
        setup_x = clamp(ball_x - 0.42 * drive_dx, -half_len + 0.55, half_len - 0.75)
        setup_y = clamp(ball_y - 0.42 * drive_dy, -half_wid + 0.28, half_wid - 0.28)
        _set_behavior(
            agent,
            "cup_vla_protect_lead_hold_repose",
            score_margin=scene.get("score_margin"),
            secs_remaining=scene.get("secs_remaining"),
            ball_dist=round(ball_dist, 3),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            behind=round(behind, 2),
            lateral=round(lateral, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return True

    hold_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
    hold_err_deg = normalize_angle_deg(hold_yaw_deg - agent.get_self_yaw())
    vx = clamp(0.92 * raw_vx + 0.05 * b_x, 0.08, 0.68)
    vy = clamp(0.96 * raw_vy + 0.14 * b_y, -0.72, 0.72)
    w = clamp(
        math.radians(hold_err_deg) * 0.54 + 0.08 * math.atan2(b_y, max(abs(b_x), 0.06)), -0.62, 0.62
    )
    _set_behavior(
        agent,
        "cup_vla_protect_lead_hold_drive",
        score_margin=scene.get("score_margin"),
        secs_remaining=scene.get("secs_remaining"),
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        raw_vx=round(raw_vx, 2),
        raw_vy=round(raw_vy, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
        hold_x=round(target_x, 2),
        hold_y=round(target_y, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)
    return True


def _cup_own_wide_open_clear(agent, tracker, ball_map):
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    if ball_x > -0.75 or abs(ball_y) < 1.45:
        tracker.zone_age("own_wide_open_clear", False)
        return False
    side = 1.0 if ball_y > 0.0 else -1.0
    trap_age = tracker.zone_age("own_wide_open_clear", True)
    if trap_age > 3.0:
        clear_target_x = clamp(ball_x + 3.65, 0.85, 2.25)
        clear_target_y = clamp(-0.26 * side, -0.38, 0.38)
    else:
        clear_target_x = clamp(ball_x + 3.05, 0.35, 1.55)
        clear_target_y = 0.0
    clear_dx = clear_target_x - ball_x
    clear_dy = clear_target_y - ball_y
    clear_norm = max(math.hypot(clear_dx, clear_dy), 1e-6)
    clear_dx /= clear_norm
    clear_dy /= clear_norm

    if not agent.get_if_ball():
        setup_x = clamp(ball_x - 0.34 * clear_dx, -half_len + 0.48, 0.25)
        setup_y = clamp(ball_y - 0.78 * side, -half_wid + 0.34, half_wid - 0.34)
        _set_behavior(
            agent,
            "cup_vla_own_wide_open_clear_setup",
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            clear_x=round(clear_target_x, 2),
            clear_y=round(clear_target_y, 2),
        )
        _drive_to_pose(
            agent, setup_x, setup_y, face_x=clear_target_x, face_y=clear_target_y, stop_dist=0.11
        )
        agent.move_head(math.inf, math.inf)
        return True

    pos = agent.get_self_pos()
    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    raw_vx, raw_vy = _map_vector_to_robot(agent, clear_dx, clear_dy)
    rel_x = pos[0] - ball_x
    rel_y = pos[1] - ball_y
    behind = rel_x * clear_dx + rel_y * clear_dy
    lateral = -rel_x * clear_dy + rel_y * clear_dx
    deep_corner_trap = ball_x < -half_len + 1.15 and abs(ball_y) > half_wid - 0.45
    ready = (
        ball_dist < 2.15
        and b_x > -1.05
        and raw_vx > -0.55
        and behind < 0.78
        and abs(lateral) < 1.75
    )
    force_clear = (trap_age > 4.0 or deep_corner_trap) and ball_dist < 2.85 and raw_vx > -0.90
    if force_clear:
        ready = True
    if not ready:
        setup_x = clamp(ball_x - 0.34 * clear_dx, -half_len + 0.48, 0.25)
        setup_y = clamp(ball_y - 0.82 * side, -half_wid + 0.34, half_wid - 0.34)
        _set_behavior(
            agent,
            "cup_vla_own_wide_open_clear_repose",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            behind=round(behind, 2),
            lateral=round(lateral, 2),
            raw_vx=round(raw_vx, 2),
            trap_age=round(trap_age, 1),
        )
        _drive_to_pose(
            agent, setup_x, setup_y, face_x=clear_target_x, face_y=clear_target_y, stop_dist=0.1
        )
        agent.move_head(math.inf, math.inf)
        return True

    clear_yaw_deg = math.degrees(math.atan2(clear_target_y - pos[1], clear_target_x - pos[0]))
    clear_err_deg = normalize_angle_deg(clear_yaw_deg - agent.get_self_yaw())
    if force_clear:
        if deep_corner_trap:
            # In deep own corners, perfect behind-ball alignment is usually
            # impossible with the gait. Scrape inward first, then let the next
            # VLA cycle choose a safer outlet.
            vx = clamp(0.42 + 0.16 * max(raw_vx, -0.2) + 0.04 * b_x, 0.20, 0.68)
            vy = clamp(-1.05 * side + 0.16 * b_y, -1.0, 1.0)
        else:
            vx = clamp(1.08 * raw_vx + 0.14 * b_x, 0.18, 1.0)
            vy = clamp(1.36 * raw_vy + 0.22 * b_y - 0.24 * side, -1.0, 1.0)
        w = clamp(
            math.radians(clear_err_deg) * 0.88 + 0.16 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.95,
            0.95,
        )
        behavior = "cup_vla_own_wide_force_clear"
    else:
        vx = clamp(1.02 * raw_vx + 0.10 * b_x, 0.04, 0.9)
        vy = clamp(1.20 * raw_vy + 0.18 * b_y - 0.38 * ball_y, -1.0, 1.0)
        w = clamp(
            math.radians(clear_err_deg) * 0.72 + 0.12 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.82,
            0.82,
        )
        behavior = "cup_vla_own_wide_open_clear_drive"
    _set_behavior(
        agent,
        behavior,
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        raw_vx=round(raw_vx, 2),
        raw_vy=round(raw_vy, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
        clear_x=round(clear_target_x, 2),
        clear_y=round(clear_target_y, 2),
        trap_age=round(trap_age, 1),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)
    return True


def _cup_high_wide_cutback_finish(agent, tracker, ball_map):
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    goal_x = half_len
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    if ball_x < 2.0 or abs(ball_y) < 1.05:
        return False
    if abs(ball_y) > half_wid - 0.06:
        return False
    if not agent.get_if_ball():
        side = 1.0 if ball_y > 0.0 else -1.0
        if ball_x > goal_x - 1.15:
            target_x = goal_x + 0.32
            target_y = side * 0.16
        else:
            target_x = clamp(ball_x + 1.25, 3.25, goal_x - 0.38)
            target_y = side * 0.42
        cut_dx = target_x - ball_x
        cut_dy = target_y - ball_y
        cut_norm = max(math.hypot(cut_dx, cut_dy), 1e-6)
        setup_x = clamp(ball_x - 0.36 * cut_dx / cut_norm, 1.75, goal_x - 0.18)
        setup_y = clamp(ball_y - 0.64 * side, -half_wid + 0.34, half_wid - 0.34)
        _set_behavior(
            agent,
            "cup_vla_high_wide_cutback_setup",
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return True

    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    if ball_dist > 1.42:
        return False
    side = 1.0 if ball_y > 0.0 else -1.0
    pos = agent.get_self_pos()
    if ball_x > goal_x - 1.15:
        target_x = goal_x + (0.46 if ball_x > goal_x - 0.8 else 0.18)
        target_y = side * (0.10 if abs(ball_y) < 1.55 else 0.20)
    else:
        target_x = clamp(ball_x + 1.35, 3.2, goal_x - 0.36)
        target_y = side * 0.40
    cut_dx = target_x - ball_x
    cut_dy = target_y - ball_y
    cut_norm = max(math.hypot(cut_dx, cut_dy), 1e-6)
    cut_dx /= cut_norm
    cut_dy /= cut_norm
    raw_vx, raw_vy = _map_vector_to_robot(agent, cut_dx, cut_dy)
    rel_x = pos[0] - ball_x
    rel_y = pos[1] - ball_y
    behind = rel_x * cut_dx + rel_y * cut_dy
    lateral = -rel_x * cut_dy + rel_y * cut_dx
    ready = (
        ball_dist < 1.58
        and b_x > -0.48
        and raw_vx > -0.30
        and behind < 0.48
        and abs(lateral) < 1.28
    )
    if not ready:
        setup_x = clamp(ball_x - 0.36 * cut_dx, 1.75, goal_x - 0.18)
        setup_y = clamp(ball_y - 0.70 * side, -half_wid + 0.34, half_wid - 0.34)
        _set_behavior(
            agent,
            "cup_vla_high_wide_cutback_repose",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            behind=round(behind, 2),
            lateral=round(lateral, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return True

    cut_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
    cut_err_deg = normalize_angle_deg(cut_yaw_deg - agent.get_self_yaw())
    vx = clamp(1.04 * raw_vx + 0.06 * b_x, 0.10, 0.86)
    vy = clamp(1.52 * raw_vy + 0.16 * b_y - 0.12 * side, -1.0, 1.0)
    w = clamp(
        math.radians(cut_err_deg) * 0.82 + 0.12 * math.atan2(b_y, max(abs(b_x), 0.06)), -0.86, 0.86
    )
    _set_behavior(
        agent,
        "cup_vla_high_wide_cutback_finish",
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        raw_vx=round(raw_vx, 2),
        raw_vy=round(raw_vy, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
        target_x=round(target_x, 2),
        target_y=round(target_y, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)
    return True


def _cup_sideline_escape_direct(agent, tracker, ball_map):
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    side = 1.0 if ball_y > 0.0 else -1.0
    trap_key = (round(ball_x * 2.0) / 2.0, side)
    now = time.time()
    if trap_key != getattr(agent, "sideline_trap_key", None):
        agent.sideline_trap_key = trap_key
        agent.sideline_trap_ts = now
    trap_age = now - getattr(agent, "sideline_trap_ts", now)
    lane_y = clamp(side * 0.36, -0.46, 0.46)
    if ball_x > half_len - 1.15:
        target_x = half_len - 0.28
        target_y = clamp(side * 0.30, -0.38, 0.38)
    elif ball_x > 1.3:
        target_x = half_len - 0.42
        target_y = lane_y
    elif abs(ball_y) > half_wid - 0.18:
        target_x = clamp(ball_x + 0.25, -half_len + 0.55, half_len - 0.9)
        target_y = clamp(side * 1.18, -1.25, 1.25)
    elif ball_x > -0.65:
        target_x = 2.55
        target_y = lane_y
    else:
        target_x = 0.75
        target_y = clamp(ball_y * 0.18, -0.62, 0.62)

    if not agent.get_if_ball():
        approach_x = clamp(ball_x - 0.42, -half_len + 0.45, half_len - 0.55)
        approach_y = clamp(ball_y - side * 0.86, -half_wid + 0.42, half_wid - 0.42)
        _set_behavior(
            agent,
            "cup_vln_sideline_inner_approach",
            target_x=round(approach_x, 2),
            target_y=round(approach_y, 2),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            trap_age=round(trap_age, 1),
        )
        _drive_to_pose(
            agent, approach_x, approach_y, face_x=target_x, face_y=target_y, stop_dist=0.12
        )
        agent.move_head(math.inf, math.inf)
        return

    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    pos = agent.get_self_pos()
    if ball_x > half_len - 0.72 and abs(ball_y) > half_wid - 0.42 and ball_dist < 1.85:
        target_x = half_len + 0.12
        target_y = clamp(-0.08 * side, -0.16, 0.16)
        sweep_dx = target_x - ball_x
        sweep_dy = target_y - ball_y
        sweep_norm = max(math.hypot(sweep_dx, sweep_dy), 1e-6)
        sweep_dx /= sweep_norm
        sweep_dy /= sweep_norm
        raw_vx, raw_vy = _map_vector_to_robot(agent, sweep_dx, sweep_dy)
        sweep_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
        sweep_err_deg = normalize_angle_deg(sweep_yaw_deg - agent.get_self_yaw())
        vx = clamp(0.62 * raw_vx + 0.03 * b_x, -0.04, 0.34)
        vy = clamp(1.65 * raw_vy + 0.20 * b_y - 0.18 * side, -1.0, 1.0)
        w = clamp(
            math.radians(sweep_err_deg) * 0.82 + 0.10 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.92,
            0.92,
        )
        _set_behavior(
            agent,
            "cup_vln_front_corner_sweep_goal",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return
    if ball_x > 1.55 and ball_dist < 1.45:
        if ball_x > half_len - 0.85:
            finish_target_x = half_len + 0.42
            finish_target_y = clamp(-0.05 * side, -0.12, 0.12)
            setup_back = 0.34
            behavior_prefix = "cup_vln_front_corner"
        else:
            finish_target_x = half_len - 0.32
            finish_target_y = clamp(side * 0.28, -0.36, 0.36)
            setup_back = 0.42
            behavior_prefix = "cup_vln_front_sideline"
        finish_dx = finish_target_x - ball_x
        finish_dy = finish_target_y - ball_y
        finish_norm = max(math.hypot(finish_dx, finish_dy), 1e-6)
        finish_dx /= finish_norm
        finish_dy /= finish_norm
        raw_vx, raw_vy = _map_vector_to_robot(agent, finish_dx, finish_dy)
        rel_x = pos[0] - ball_x
        rel_y = pos[1] - ball_y
        behind = rel_x * finish_dx + rel_y * finish_dy
        lateral = -rel_x * finish_dy + rel_y * finish_dx
        front_ready = (
            b_x > -0.18
            and ball_dist < 1.18
            and raw_vx > -0.10
            and behind < 0.24
            and abs(lateral) < 0.9
        )
        if not front_ready and trap_age < 9.0:
            setup_x = clamp(ball_x - setup_back * finish_dx, 1.1, half_len - 0.25)
            setup_y = clamp(ball_y - side * 0.78, -half_wid + 0.36, half_wid - 0.36)
            _set_behavior(
                agent,
                f"{behavior_prefix}_setup",
                ball_dist=round(ball_dist, 3),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                behind=round(behind, 2),
                lateral=round(lateral, 2),
                raw_vx=round(raw_vx, 2),
                trap_age=round(trap_age, 1),
            )
            _drive_to_pose(
                agent,
                setup_x,
                setup_y,
                face_x=finish_target_x,
                face_y=finish_target_y,
                stop_dist=0.11,
            )
            agent.move_head(math.inf, math.inf)
            return
        if not front_ready:
            # If the front-corner skill is stuck, stop trying to perfect the pose:
            # scrape the ball back into the central lane and let the second robot finish.
            finish_target_x = clamp(ball_x - 1.25, 1.65, half_len - 0.85)
            finish_target_y = clamp(side * 0.18, -0.28, 0.28)
            finish_dx = finish_target_x - ball_x
            finish_dy = finish_target_y - ball_y
            finish_norm = max(math.hypot(finish_dx, finish_dy), 1e-6)
            finish_dx /= finish_norm
            finish_dy /= finish_norm
            raw_vx, raw_vy = _map_vector_to_robot(agent, finish_dx, finish_dy)
        finish_yaw_deg = math.degrees(
            math.atan2(finish_target_y - pos[1], finish_target_x - pos[0])
        )
        finish_err_deg = normalize_angle_deg(finish_yaw_deg - agent.get_self_yaw())
        vx = clamp(1.08 * raw_vx + 0.08 * b_x, 0.18, 0.88)
        if trap_age > 9.0:
            vx = clamp(0.86 * raw_vx + 0.04 * b_x, -0.06, 0.62)
        vy = clamp(1.18 * raw_vy + 0.22 * b_y - 0.18 * side, -1.0, 1.0)
        w = clamp(
            math.radians(finish_err_deg) * 0.62 + 0.12 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.78,
            0.78,
        )
        _set_behavior(
            agent,
            f"{behavior_prefix}_drive" if trap_age <= 9.0 else f"{behavior_prefix}_release",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
            target_x=round(finish_target_x, 2),
            target_y=round(finish_target_y, 2),
            trap_age=round(trap_age, 1),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    if ball_x < -0.85 and ball_dist < 1.22:
        clear_target_x = 1.25
        clear_target_y = clamp(-0.18 * side, -0.34, 0.34)
        clear_dx = clear_target_x - ball_x
        clear_dy = clear_target_y - ball_y
        clear_norm = max(math.hypot(clear_dx, clear_dy), 1e-6)
        clear_dx /= clear_norm
        clear_dy /= clear_norm
        raw_vx, raw_vy = _map_vector_to_robot(agent, clear_dx, clear_dy)
        clear_yaw_deg = math.degrees(math.atan2(clear_target_y - pos[1], clear_target_x - pos[0]))
        clear_err_deg = normalize_angle_deg(clear_yaw_deg - agent.get_self_yaw())
        if raw_vx < -0.12 or b_x < -0.2:
            setup_x = clamp(ball_x - 0.36 * clear_dx, -half_len + 0.45, half_len - 0.5)
            setup_y = clamp(ball_y - 0.58 * side, -half_wid + 0.38, half_wid - 0.38)
            _set_behavior(
                agent,
                "cup_vln_def_sideline_clear_setup",
                ball_dist=round(ball_dist, 3),
                target_x=round(setup_x, 2),
                target_y=round(setup_y, 2),
                raw_vx=round(raw_vx, 2),
                trap_age=round(trap_age, 1),
            )
            _drive_to_pose(
                agent,
                setup_x,
                setup_y,
                face_x=clear_target_x,
                face_y=clear_target_y,
                stop_dist=0.12,
            )
            agent.move_head(math.inf, math.inf)
            return
        vx = clamp(1.15 * raw_vx + 0.08 * b_x, 0.42, 1.0)
        vy = clamp(1.08 * raw_vy + 0.18 * b_y - 0.25 * side, -1.0, 1.0)
        w = clamp(
            math.radians(clear_err_deg) * 0.58 + 0.12 * math.atan2(b_y, max(b_x, 0.04)), -0.78, 0.78
        )
        _set_behavior(
            agent,
            "cup_vln_def_sideline_scrape_clear",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            raw_vx=round(raw_vx, 2),
            raw_vy=round(raw_vy, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
            target_x=round(clear_target_x, 2),
            target_y=round(clear_target_y, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    inside_gap = side * (ball_y - pos[1])
    need_inner_setup = inside_gap < 0.42 or ball_dist > 1.18
    if trap_age > 2.0 and inside_gap > 0.75 and ball_dist < 2.25:
        # Skill-level recovery: once we are safely inside the touchline, stop
        # optimizing the approach pose and scrape the ball back into the field.
        need_inner_setup = False
    if trap_age > 5.0 and ball_dist < 1.65:
        need_inner_setup = False
    if need_inner_setup:
        approach_x = clamp(ball_x - 0.48, -half_len + 0.45, half_len - 0.55)
        approach_y = clamp(ball_y - side * 0.92, -half_wid + 0.42, half_wid - 0.42)
        _set_behavior(
            agent,
            "cup_vln_sideline_inner_setup",
            ball_dist=round(ball_dist, 3),
            target_x=round(approach_x, 2),
            target_y=round(approach_y, 2),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            inside_gap=round(inside_gap, 2),
            trap_age=round(trap_age, 1),
        )
        _drive_to_pose(
            agent, approach_x, approach_y, face_x=target_x, face_y=target_y, stop_dist=0.14
        )
        agent.move_head(math.inf, math.inf)
        return

    desired_x = target_x - ball_x
    desired_y = target_y - ball_y
    desired_norm = max(math.hypot(desired_x, desired_y), 1e-6)
    desired_x /= desired_norm
    desired_y /= desired_norm
    raw_vx, raw_vy = _map_vector_to_robot(agent, desired_x, desired_y)
    drive_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
    drive_err_deg = normalize_angle_deg(drive_yaw_deg - agent.get_self_yaw())

    if raw_vx < -0.16 or b_x < -0.18:
        vx = clamp(0.12 + 0.18 * max(b_x, -0.2), -0.04, 0.24)
        vy = clamp(1.05 * raw_vy + 0.24 * b_y - 0.18 * side, -1.0, 1.0)
        w = clamp(
            math.radians(drive_err_deg) * 0.95 + 0.12 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.9,
            0.9,
        )
        behavior = "cup_vln_sideline_reorient"
    else:
        vx = clamp(1.05 * raw_vx + 0.06 * b_x, 0.34, 0.9)
        vy = clamp(1.05 * raw_vy + 0.14 * b_y - 0.12 * side, -1.0, 1.0)
        w = clamp(
            math.radians(drive_err_deg) * 0.62 + 0.08 * math.atan2(b_y, max(b_x, 0.04)), -0.72, 0.72
        )
        behavior = "cup_vln_sideline_infield_drive"
    if abs(ball_y) > half_wid - 0.22:
        vy = clamp(vy - 0.55 * side, -1.0, 1.0)
        vx = clamp(vx, -0.02, 0.42)

    _set_behavior(
        agent,
        behavior,
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        raw_vx=round(raw_vx, 2),
        raw_vy=round(raw_vy, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
        target_x=round(target_x, 2),
        target_y=round(target_y, 2),
        trap_age=round(trap_age, 1),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)


def _cup_offball(agent, tracker, ball_map, handler_id):
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    goal_x = half_len
    own_goal_x = -half_len
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])

    if agent.role == "keeper":
        _cup_keeper(agent, tracker, ball_map)
        return

    score_margin = int(agent.gamecontroller.score) - int(agent.gamecontroller.opponent_score)
    secs_remaining = getattr(agent.gamecontroller, "secs_remaining", 600)
    try:
        secs_remaining = int(secs_remaining)
    except (TypeError, ValueError):
        secs_remaining = 600
    if score_margin > 0 and secs_remaining <= 120:
        side = 1.0 if ball_y >= 0.0 else -1.0
        if agent.role == "support":
            target_x = clamp(ball_x - 1.1, -1.75, 0.85)
            target_y = clamp(ball_y * 0.18, -0.45, 0.45)
            behavior = "cup_vla_protect_lead_center_screen"
        else:
            target_x = clamp(ball_x - 0.45, -0.65, 1.45)
            target_y = clamp(-side * 0.85, -1.05, 1.05)
            behavior = "cup_vla_protect_lead_weakside_screen"
        _set_behavior(
            agent,
            behavior,
            score_margin=score_margin,
            secs_remaining=secs_remaining,
            handler=handler_id,
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.14)
        return

    if abs(ball_y) > half_wid - 0.62:
        side = 1.0 if ball_y > 0.0 else -1.0
        if ball_x > goal_x - 1.05:
            target_x = clamp(goal_x - 0.82, 3.1, goal_x - 0.45)
            target_y = clamp(-side * 0.20, -0.36, 0.36)
            behavior = "cup_vln_corner_goal_crash_receiver"
        elif ball_x > 1.5:
            target_x = clamp(ball_x + 0.35, 2.15, goal_x - 0.65)
            target_y = clamp(-side * 0.28, -0.46, 0.46)
            behavior = "cup_vln_sideline_cutback_receiver"
        elif ball_x > -0.6:
            target_x = clamp(ball_x + 0.95, 0.4, 2.65)
            target_y = clamp(side * 0.36, -0.52, 0.52)
            behavior = "cup_vln_sideline_infield_receiver"
        else:
            target_x = clamp(ball_x + 1.65, -0.2, 1.15)
            target_y = clamp(-side * 0.42, -0.58, 0.58)
            behavior = "cup_vln_sideline_counter_receiver"
    elif ball_x < own_goal_x + 1.35:
        target_x = clamp(ball_x + 2.25, -0.35, 0.85)
        target_y = clamp(-0.42 if ball_y >= 0.0 else 0.42, -0.58, 0.58)
        behavior = "cup_deep_counter_outlet"
    elif ball_x < -0.8:
        target_x = clamp(ball_x + 1.45, -0.15, 1.25)
        target_y = clamp(-0.65 if ball_y >= 0.0 else 0.65, -0.9, 0.9)
        behavior = "cup_counter_outlet"
    elif ball_x < 1.8:
        target_x = clamp(ball_x + 1.35, 1.25, 2.75)
        target_y = clamp(ball_y * 0.16, -0.45, 0.45)
        behavior = "cup_front_receiver"
    else:
        target_x = clamp(goal_x - 0.5, 3.2, goal_x - 0.35)
        target_y = clamp(-0.38 if ball_y >= 0.0 else 0.38, -0.52, 0.52)
        behavior = "cup_goalmouth_receiver"

    _set_behavior(
        agent,
        behavior,
        handler=handler_id,
        target_x=round(target_x, 2),
        target_y=round(target_y, 2),
    )
    _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.14)


def _cup_goal_kick_executor_id(tracker, ball_map):
    candidates = []
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    for local_id, info in tracker.players.items():
        if info["penalty"] != 0 or info["role"] == "keeper":
            continue
        px, py = info["pos"]
        dist = math.hypot(ball_x - px, ball_y - py)
        lateral = abs(ball_y - py)
        score = dist + 0.32 * lateral
        if info["role"] == "support":
            score -= 0.75
        elif ball_x < -tracker.field_length / 2.0 + 1.0:
            score += 0.28
        if lateral > 1.35:
            score += 0.55
        if px > ball_x + 0.55:
            score += 0.18
        candidates.append((score, local_id))
    if not candidates:
        return None
    return min(candidates)[1]


def _cup_own_goal_kick(agent, tracker, ball_map):
    half_len = agent.field_length / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
    score_margin = int(agent.gamecontroller.score) - int(agent.gamecontroller.opponent_score)
    secs_remaining = getattr(agent.gamecontroller, "secs_remaining", 600)
    try:
        secs_remaining = int(secs_remaining)
    except (TypeError, ValueError):
        secs_remaining = 600
    protect_restart = score_margin > 0 and secs_remaining <= 120

    if agent.role == "keeper":
        guard_x = -half_len + 0.46
        guard_y = clamp(ball_y * 0.18, -0.28, 0.28)
        _set_behavior(
            agent,
            "cup_goal_kick_keeper_line",
            target_x=round(guard_x, 2),
            target_y=round(guard_y, 2),
        )
        _drive_to_pose(agent, guard_x, guard_y, face_x=ball_x + 1.6, face_y=0.0, stop_dist=0.1)
        return

    executor_id = _cup_goal_kick_executor_id(tracker, ball_map)
    if executor_id is None:
        _set_behavior(agent, "cup_goal_kick_no_executor")
        agent.stop()
        return

    if executor_id != local_id:
        side = -1.0 if agent.role == "striker" else 1.0
        target_x = clamp(ball_x + 2.7, -1.2, -0.2)
        target_y = clamp(side * 0.62, -0.78, 0.78)
        _set_behavior(
            agent,
            "cup_goal_kick_second_ball",
            handler=executor_id,
            target_x=round(target_x, 2),
            target_y=round(target_y, 2),
        )
        _drive_to_pose(agent, target_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.16)
        return

    side = -1.0 if ball_y >= 0.0 else 1.0
    if protect_restart:
        target_x = clamp(ball_x + 2.35, -1.45, 0.18)
        target_y = clamp(ball_y * 0.08, -0.22, 0.22)
    else:
        target_x = clamp(ball_x + 3.05, -1.05, 0.72)
        target_y = clamp(side * 0.48 + ball_y * 0.06, -0.62, 0.62)
    push_dx = target_x - ball_x
    push_dy = target_y - ball_y
    push_norm = max(math.hypot(push_dx, push_dy), 1e-6)
    push_dx /= push_norm
    push_dy /= push_norm
    setup_back = 0.22 if ball_x < -half_len + 0.62 else 0.38
    setup_x = clamp(ball_x - setup_back * push_dx, -half_len + 0.36, -3.38)
    setup_y = clamp(ball_y - setup_back * push_dy, -0.58, 0.58)

    if not agent.get_if_ball():
        _set_behavior(
            agent,
            "cup_goal_kick_executor_approach",
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.12)
        agent.move_head(math.inf, math.inf)
        return

    pos = agent.get_self_pos()
    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)
    raw_vx, raw_vy = _map_vector_to_robot(agent, push_dx, push_dy)
    rel_x = pos[0] - ball_x
    rel_y = pos[1] - ball_y
    behind = rel_x * push_dx + rel_y * push_dy
    lateral = -rel_x * push_dy + rel_y * push_dx

    ready = (
        ball_dist < 0.78
        and b_x > -0.16
        and abs(b_y) < 0.46
        and raw_vx > -0.04
        and behind < 0.18
        and abs(lateral) < 0.72
    )
    emergency_ready = ball_dist < 2.1 and b_x > -0.55 and raw_vx > -0.28
    if not ready and emergency_ready:
        clear_err_deg = normalize_angle_deg(
            math.degrees(math.atan2(target_y - pos[1], target_x - pos[0])) - agent.get_self_yaw()
        )
        if protect_restart:
            vx = clamp(0.88 * raw_vx + 0.06 * b_x, 0.14, 0.62)
            vy = clamp(0.70 * raw_vy + 0.14 * b_y - 0.28 * ball_y, -0.46, 0.46)
        else:
            vx = clamp(1.05 * raw_vx + 0.08 * b_x, 0.22, 0.86)
            vy = clamp(1.05 * raw_vy + 0.20 * b_y, -0.88, 0.88)
        w = clamp(
            math.radians(clear_err_deg) * 0.68 + 0.12 * math.atan2(b_y, max(abs(b_x), 0.06)),
            -0.78,
            0.78,
        )
        _set_behavior(
            agent,
            "cup_goal_kick_protect_clear" if protect_restart else "cup_goal_kick_emergency_clear",
            ball_dist=round(ball_dist, 3),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            lateral=round(lateral, 2),
            raw_vx=round(raw_vx, 2),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return
    if not ready:
        _set_behavior(
            agent,
            "cup_goal_kick_executor_setup",
            ball_dist=round(ball_dist, 3),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            behind=round(behind, 2),
            lateral=round(lateral, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=target_x, face_y=target_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return

    clear_yaw_deg = math.degrees(math.atan2(target_y - pos[1], target_x - pos[0]))
    clear_err_deg = normalize_angle_deg(clear_yaw_deg - agent.get_self_yaw())
    if protect_restart:
        vx = clamp(0.90 * raw_vx + 0.05 * b_x, 0.22, 0.62)
        vy = clamp(0.56 * raw_vy + 0.10 * b_y - 0.24 * ball_y, -0.34, 0.34)
    else:
        vx = clamp(1.05 * raw_vx + 0.06 * b_x, 0.46, 0.88)
        vy = clamp(0.85 * raw_vy + 0.12 * b_y, -0.42, 0.42)
    w = clamp(
        math.radians(clear_err_deg) * 0.52 + 0.16 * math.atan2(b_y, max(b_x, 0.04)), -0.5, 0.5
    )
    _set_behavior(
        agent,
        "cup_goal_kick_protect_short_clear" if protect_restart else "cup_goal_kick_clear",
        ball_dist=round(ball_dist, 3),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)


def _cup_keeper(agent, tracker, ball_map):
    half_len = agent.field_length / 2.0
    own_goal_x = -half_len
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])

    if (
        agent.get_if_ball()
        and ball_x < own_goal_x + 0.62
        and abs(ball_y) < 1.18
        and agent.get_ball_distance() < 0.92
    ):
        _set_behavior(agent, "cup_keeper_last_clear", ball_dist=round(agent.get_ball_distance(), 3))
        _keeper_clear_ball(agent, tracker, ball_map)
        return

    guard_x = own_goal_x + 0.72
    if ball_x < own_goal_x + 1.35:
        guard_x = own_goal_x + 0.95
    target_y = clamp(ball_y * 0.52, -0.72, 0.72)
    _set_behavior(
        agent, "cup_keeper_guard", target_x=round(guard_x, 2), target_y=round(target_y, 2)
    )
    _drive_to_pose(agent, guard_x, target_y, face_x=ball_x, face_y=ball_y, stop_dist=0.1)


def _cup_own_restart(agent, tracker, ball_map):
    if agent.gamecontroller.set_play_name == "SET_PLAY_GOAL_KICK":
        _cup_own_goal_kick(agent, tracker, ball_map)
        return

    if agent.role == "keeper":
        _cup_keeper(agent, tracker, ball_map)
        return

    gc = agent.gamecontroller
    local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
    handler_id = _cup_nonkeeper_handler_id(tracker, ball_map, prefer_support=True)
    if handler_id == local_id:
        _cup_drive_ball(agent, tracker, ball_map, gc.set_play_name)
        return
    _cup_offball(agent, tracker, ball_map, handler_id)


def _cup_open_play(agent, tracker, ball_map):
    if agent.role == "keeper":
        _cup_keeper(agent, tracker, ball_map)
        return

    local_id = int(getattr(agent, "local_id", agent.get_config().get("id", 0)))
    handler_id = _cup_nonkeeper_handler_id(tracker, ball_map)
    if handler_id == local_id:
        _cup_drive_ball(agent, tracker, ball_map)
        return
    _cup_offball(agent, tracker, ball_map, handler_id)


def _cup_match_controller(agent, tracker):
    ball_map = _get_team_ball(agent, tracker)
    if ball_map is None:
        if agent.role == "keeper":
            _keeper_anchor(agent, tracker)
        else:
            _set_behavior(agent, "cup_find_ball")
            agent.state_machine_runners["find_ball"]()
        return

    gc = agent.gamecontroller
    if gc.set_play_name not in ("SET_PLAY_NONE", None) and not gc.can_kick:
        _opponent_setplay_defense(agent, tracker)
        return
    if (
        gc.set_play_name
        in ("SET_PLAY_GOAL_KICK", "SET_PLAY_KICK_IN", "SET_PLAY_CORNER_KICK", "SET_PLAY_KICK_OFF")
        and gc.can_kick
    ):
        _cup_own_restart(agent, tracker, ball_map)
        return
    _cup_open_play(agent, tracker, ball_map)


def _champion_game(agent, tracker):
    gc = agent.gamecontroller
    tracker.choose_primary()

    if gc.penalty != 0:
        _set_behavior(agent, "penalized_stop", penalty=gc.player_penalty_name)
        agent.stop()
        return

    if gc.game_state == "STATE_INITIAL":
        _set_behavior(agent, "initial_stop")
        agent.stop()
        return

    if gc.game_state == "STATE_READY":
        tx, ty, tyaw = _get_setplay_ready_pose(agent, tracker)
        _set_behavior(agent, "ready_pose", target_x=round(tx, 2), target_y=round(ty, 2))
        _drive_to_pose(agent, tx, ty, face_yaw_deg=tyaw, stop_dist=0.15)
        return

    if gc.game_state == "STATE_SET":
        _set_behavior(agent, "set_stop")
        agent.stop()
        return

    if gc.game_state in ("STATE_FINISHED", "STATE_STANDBY"):
        _set_behavior(agent, "finished_stop")
        agent.stop()
        return

    if gc.game_state != "STATE_PLAYING":
        _set_behavior(agent, "unknown_stop", state=gc.game_state)
        agent.stop()
        return

    if (
        agent.role == "keeper"
        and gc.set_play_name == "SET_PLAY_KICK_OFF"
        and gc.secs_remaining is not None
        and int(gc.secs_remaining) >= 588
    ):
        ball_map = _get_team_ball(agent, tracker)
        if ball_map is not None:
            _cup_keeper(agent, tracker, ball_map)
        else:
            _keeper_anchor(agent, tracker)
        return

    if _boundary_recovery(agent, tracker):
        return

    _cup_match_controller(agent, tracker)


def _penalty_attack_sign(agent):
    return -1.0 if str(getattr(agent, "color", "red")).lower() == "blue" else 1.0


def _penalty_goal_target(agent, ball_map=None):
    attack = _penalty_attack_sign(agent)
    half_len = agent.field_length / 2.0
    goal_x = attack * half_len
    goal_y = 0.0
    if ball_map is not None:
        # If the ball is already wide, aim slightly inside the goal corridor
        # instead of pushing along the sideline.
        ball_y = float(ball_map[1])
        goal_y = clamp(-0.42 * ball_y, -0.58, 0.58)
        if abs(ball_y) > 1.35:
            goal_x = attack * (half_len - 0.72)
    return goal_x, goal_y


def _penalty_default_ball(agent):
    attack = _penalty_attack_sign(agent)
    half_len = agent.field_length / 2.0
    # Official qualifier spot is 1.5m from the target goal line on S-Field.
    return attack * (half_len - 1.5), 0.0


def _penalty_setup_pose(agent, ball_map, setup_back=0.58):
    attack = _penalty_attack_sign(agent)
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    goal_x, goal_y = _penalty_goal_target(agent, ball_map)
    dx = goal_x - ball_x
    dy = goal_y - ball_y
    norm = max(math.hypot(dx, dy), 1e-6)
    ux = dx / norm
    uy = dy / norm
    setup_x = ball_x - setup_back * ux
    setup_y = ball_y - setup_back * uy
    if abs(ball_y) > half_wid - 0.65:
        setup_x = ball_x - setup_back * attack
        setup_y = ball_y * 0.62
    elif abs(ball_y) > 1.35:
        setup_x = ball_x - setup_back * attack
        setup_y = ball_y * 0.70
    # Keep enough margin from boundaries for the gait policy to turn and settle.
    setup_x = clamp(setup_x, -half_len + 0.45, half_len - 0.45)
    setup_y = clamp(setup_y, -half_wid + 0.22, half_wid - 0.22)
    face_yaw = 0.0 if attack > 0.0 else 180.0
    return setup_x, setup_y, face_yaw


def _penalty_ready_pose(agent, tracker=None):
    half_len = agent.field_length / 2.0
    attack = _penalty_attack_sign(agent)
    ball_map = _get_team_ball(agent, tracker) if tracker is not None else None
    if ball_map is None:
        ball_map = _penalty_default_ball(agent)
    striker_x, striker_y, striker_yaw = _penalty_setup_pose(agent, ball_map, setup_back=0.68)
    support_x = clamp(striker_x - attack * 0.75, -half_len + 0.55, half_len - 0.55)
    poses = {
        "striker": (striker_x, striker_y, striker_yaw),
        "support": (support_x, 0.85, striker_yaw),
        "sweeper": (support_x, -0.85, striker_yaw),
        "keeper": (-attack * half_len + attack * 0.55, 0.0, striker_yaw),
    }
    return poses.get(agent.role, (-2.0, 0.0, 0.0))


def _penalty_support_hold(agent, ball_map):
    tx, ty, tyaw = _penalty_ready_pose(agent, agent.team_tracker)
    if ball_map is not None:
        attack = _penalty_attack_sign(agent)
        bx, by = float(ball_map[0]), float(ball_map[1])
        side = 1.0 if agent.role == "support" else -1.0
        tx = clamp(
            bx - attack * 1.1, -agent.field_length / 2.0 + 0.55, agent.field_length / 2.0 - 0.55
        )
        ty = clamp(
            by + side * 0.95, -agent.field_width / 2.0 + 0.35, agent.field_width / 2.0 - 0.35
        )
    _set_behavior(agent, "penalty_mode_lane_hold", target_x=round(tx, 2), target_y=round(ty, 2))
    if ball_map is not None:
        _drive_to_pose(agent, tx, ty, face_x=ball_map[0], face_y=ball_map[1], stop_dist=0.16)
    else:
        _drive_to_pose(agent, tx, ty, face_yaw_deg=tyaw, stop_dist=0.16)


def _penalty_boundary_recovery(agent, tracker, ball_map):
    if ball_map is not None:
        attack = _penalty_attack_sign(agent)
        half_len = agent.field_length / 2.0
        ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
        scoring_lane = attack * ball_x > half_len - 2.8 and abs(ball_y) < 2.1
        if scoring_lane and agent.get_self_pos() is not None:
            pos = agent.get_self_pos()
            if len(pos) >= 2:
                x, y = float(pos[0]), float(pos[1])
                half_wid = agent.field_width / 2.0
                hard_out_x = x < -half_len - 0.16 or x > half_len + 0.16
                hard_out_y = y < -half_wid - 0.16 or y > half_wid + 0.16
                far_out = abs(y) > half_wid + 0.65 or abs(x) > half_len + 0.55
                if not (hard_out_x or hard_out_y or far_out):
                    return False
    return _boundary_recovery(agent, tracker)


def _penalty_drive_shot(agent, tracker, ball_map):
    if ball_map is None:
        _set_behavior(agent, "penalty_mode_find_ball")
        agent.state_machine_runners["find_ball"]()
        return

    attack = _penalty_attack_sign(agent)
    half_len = agent.field_length / 2.0
    half_wid = agent.field_width / 2.0
    ball_x, ball_y = float(ball_map[0]), float(ball_map[1])
    goal_x, goal_y = _penalty_goal_target(agent, ball_map)
    setup_x, setup_y, setup_yaw = _penalty_setup_pose(agent, ball_map)

    if not agent.get_if_ball():
        _set_behavior(
            agent,
            "penalty_mode_acquire_setup",
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=goal_x, face_y=goal_y, stop_dist=0.12)
        agent.move_head(math.inf, math.inf)
        return

    pos = agent.get_self_pos()
    b_x, b_y = [float(v) for v in agent.get_ball_pos()]
    ball_dist = math.hypot(b_x, b_y)

    push_dx = goal_x - ball_x
    push_dy = goal_y - ball_y
    push_norm = max(math.hypot(push_dx, push_dy), 1e-6)
    push_dx /= push_norm
    push_dy /= push_norm
    rel_x = float(pos[0]) - ball_x
    rel_y = float(pos[1]) - ball_y
    behind = rel_x * push_dx + rel_y * push_dy
    lateral = -rel_x * push_dy + rel_y * push_dx
    raw_vx, raw_vy = _map_vector_to_robot(agent, push_dx, push_dy)
    goal_yaw = math.degrees(math.atan2(goal_y - pos[1], goal_x - pos[0]))
    yaw_err = normalize_angle_deg(goal_yaw - agent.get_self_yaw())

    finish_zone = attack * ball_x > half_len - 1.2 and abs(ball_y) < 1.45

    spot_corridor = (
        attack * ball_x > half_len - 2.85
        and attack * ball_x < half_len - 0.34
        and abs(ball_y) < 1.65
    )
    if spot_corridor and ball_dist < 1.22:
        if b_x > 0.18 and abs(ball_y) < 0.38:
            vx = 0.82
        elif b_x > -0.16:
            vx = 0.42
        else:
            vx = 0.18
        vy = clamp(0.14 * b_y - 1.18 * ball_y + 0.03 * raw_vy, -0.64, 0.64)
        w = clamp(math.radians(yaw_err) * 0.06 - 0.18 * ball_y, -0.24, 0.24)
        _set_behavior(
            agent,
            "penalty_mode_spot_direct_push",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            yaw_err=round(yaw_err, 1),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    if attack * ball_x > half_len - 1.45 and abs(ball_y) > 0.68 and ball_dist < 1.38:
        cut_x = clamp(ball_x - attack * 0.68, -half_len + 0.45, half_len - 0.45)
        cut_y = clamp(ball_y * 0.22, -0.52, 0.52)
        if b_x > 0.02 and abs(b_y) < 0.36 and abs(yaw_err) < 70.0:
            cut_dx = attack * half_len - ball_x
            cut_dy = -1.15 * ball_y
            cut_norm = max(math.hypot(cut_dx, cut_dy), 1e-6)
            cut_raw_vx, cut_raw_vy = _map_vector_to_robot(
                agent, cut_dx / cut_norm, cut_dy / cut_norm
            )
            vx = clamp(0.78 * cut_raw_vx + 0.10, 0.18, 0.68)
            vy = clamp(1.04 * cut_raw_vy + 0.20 * b_y - 0.20 * ball_y, -0.72, 0.72)
            w = clamp(math.radians(yaw_err) * 0.28, -0.40, 0.40)
            _set_behavior(
                agent,
                "penalty_mode_goalmouth_cut_push",
                ball_dist=round(ball_dist, 3),
                ball_x=round(ball_x, 2),
                ball_y=round(ball_y, 2),
                local_x=round(b_x, 2),
                local_y=round(b_y, 2),
                yaw_err=round(yaw_err, 1),
                cmd_x=round(vx, 2),
                cmd_y=round(vy, 2),
            )
            agent.cmd_vel(vx, vy, w)
            agent.move_head(math.inf, math.inf)
            return
        _set_behavior(
            agent,
            "penalty_mode_goalmouth_cutback",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(cut_x, 2),
            target_y=round(cut_y, 2),
            yaw_err=round(yaw_err, 1),
        )
        _drive_to_pose(agent, cut_x, cut_y, face_x=goal_x, face_y=0.0, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return

    side_misaligned = ball_dist < 0.92 and abs(b_y) > 0.44 and not finish_zone
    if side_misaligned:
        safe_x = clamp(ball_x - attack * 0.66, -half_len + 0.45, half_len - 0.45)
        if attack * ball_x > half_len - 1.45 and abs(ball_y) > 0.68:
            safe_y = clamp(ball_y * 0.22, -0.52, 0.52)
        elif abs(ball_y) > 1.05:
            safe_y = clamp(ball_y * 0.42, -half_wid + 0.35, half_wid - 0.35)
        else:
            safe_y = clamp(ball_y, -half_wid + 0.35, half_wid - 0.35)
        _set_behavior(
            agent,
            "penalty_mode_no_touch_realign",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(safe_x, 2),
            target_y=round(safe_y, 2),
            lateral=round(lateral, 2),
        )
        _drive_to_pose(agent, safe_x, safe_y, face_x=goal_x, face_y=goal_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return

    if ball_dist < 1.45 and abs(lateral) > 0.82 and attack * ball_x < half_len - 1.05:
        safe_x = clamp(ball_x - attack * 0.62, -half_len + 0.45, half_len - 0.45)
        if attack * ball_x > half_len - 1.45 and abs(ball_y) > 0.68:
            safe_y = clamp(ball_y * 0.22, -0.52, 0.52)
        elif abs(ball_y) > 1.0:
            safe_y = clamp(ball_y * 0.42, -half_wid + 0.35, half_wid - 0.35)
        else:
            safe_y = clamp(
                ball_y - clamp(0.05 * lateral, -0.10, 0.10), -half_wid + 0.35, half_wid - 0.35
            )
        _set_behavior(
            agent,
            "penalty_mode_recenter_setup",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(safe_x, 2),
            target_y=round(safe_y, 2),
            lateral=round(lateral, 2),
        )
        _drive_to_pose(agent, safe_x, safe_y, face_x=goal_x, face_y=goal_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return

    if finish_zone and ball_dist < 1.05:
        vx = 0.98
        vy = clamp(0.22 * b_y - 0.22 * ball_y + 0.08 * raw_vy, -0.42, 0.42)
        w = clamp(math.radians(yaw_err) * 0.28, -0.36, 0.36)
        _set_behavior(
            agent,
            "penalty_mode_finish_drive",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            yaw_err=round(yaw_err, 1),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    if ball_dist < 0.66 and b_x < -0.05:
        recover_x, recover_y, _ = _penalty_setup_pose(agent, ball_map, setup_back=0.62)
        if abs(ball_y) > 1.0:
            recover_y = clamp(ball_y * 0.42, -half_wid + 0.35, half_wid - 0.35)
        _set_behavior(
            agent,
            "penalty_mode_behind_ball_recover",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            behind=round(behind, 2),
            lateral=round(lateral, 2),
            target_x=round(recover_x, 2),
            target_y=round(recover_y, 2),
        )
        _drive_to_pose(agent, recover_x, recover_y, face_x=goal_x, face_y=goal_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return

    if ball_dist < 0.78 and b_x > -0.04 and (abs(b_y) < 0.48 or finish_zone):
        centered = abs(b_y) < 0.22 and abs(yaw_err) < 28.0
        vx = 0.82 if centered else 0.34
        if b_x < 0.08:
            vx = min(vx, 0.22)
        vy = clamp(0.82 * b_y - 0.18 * ball_y + 0.16 * raw_vy, -0.58, 0.58)
        w = clamp(
            math.radians(yaw_err) * 0.44 + 0.08 * math.atan2(b_y, max(abs(b_x), 0.05)), -0.55, 0.55
        )
        if abs(ball_y) > 1.25:
            vx = min(vx, 0.38)
            vy = clamp(vy - 0.18 * ball_y, -0.62, 0.62)
            w = clamp(math.radians(yaw_err) * 0.30, -0.42, 0.42)
        if attack * ball_x > half_len - 1.15:
            vx = max(vx, 0.62)
            vy = clamp(0.48 * vy - 0.34 * ball_y, -0.62, 0.62)
        _set_behavior(
            agent,
            "penalty_mode_contact_align",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            yaw_err=round(yaw_err, 1),
            cmd_x=round(vx, 2),
            cmd_y=round(vy, 2),
        )
        agent.cmd_vel(vx, vy, w)
        agent.move_head(math.inf, math.inf)
        return

    in_goal_corridor = abs(ball_y) < 0.74
    shot_ready = (
        ball_dist < 0.86
        and b_x > 0.05
        and abs(b_y) < 0.42
        and raw_vx > -0.08
        and behind < 0.02
        and abs(lateral) < 0.52
        and abs(yaw_err) < 34.0
    )
    # Near the target goal, prioritize a continuous body push over perfect
    # re-alignment; stopping here often wastes the high-frequency reset window.
    if attack * ball_x > half_len - 1.25 and ball_dist < 1.08 and b_x > 0.0 and abs(b_y) < 0.58:
        shot_ready = True
    if attack * ball_x > half_len - 2.05 and ball_dist < 1.45 and abs(ball_y) < 1.25:
        shot_ready = True
    if attack * ball_x > half_len - 1.85 and ball_dist < 1.58 and abs(ball_y) < 1.45:
        shot_ready = True

    if not shot_ready:
        if abs(ball_y) > half_wid - 0.55:
            setup_x = clamp(ball_x - attack * 0.44, -half_len + 0.45, half_len - 0.45)
            setup_y = clamp(ball_y * 0.48, -half_wid + 0.35, half_wid - 0.35)
        elif abs(ball_y) > 1.35:
            setup_x = clamp(ball_x - attack * 0.42, -half_len + 0.45, half_len - 0.45)
            setup_y = clamp(ball_y * 0.42, -half_wid + 0.35, half_wid - 0.35)
        _set_behavior(
            agent,
            "penalty_mode_align_goal",
            ball_dist=round(ball_dist, 3),
            ball_x=round(ball_x, 2),
            ball_y=round(ball_y, 2),
            local_x=round(b_x, 2),
            local_y=round(b_y, 2),
            target_x=round(setup_x, 2),
            target_y=round(setup_y, 2),
            behind=round(behind, 2),
            lateral=round(lateral, 2),
        )
        _drive_to_pose(agent, setup_x, setup_y, face_x=goal_x, face_y=goal_y, stop_dist=0.1)
        agent.move_head(math.inf, math.inf)
        return

    lane_pull = -0.48 * ball_y if in_goal_corridor else -0.82 * ball_y
    if attack * ball_x > half_len - 1.45:
        lane_pull = -0.28 * ball_y if abs(ball_y) < 1.25 else -0.36 * ball_y
    vx = clamp(1.10 * raw_vx + 0.10 * max(b_x, -0.2), 0.28, 1.0)
    vy = clamp(1.02 * raw_vy + 0.18 * b_y + lane_pull, -0.92, 0.92)
    w = clamp(
        math.radians(yaw_err) * 0.62 + 0.10 * math.atan2(b_y, max(abs(b_x), 0.05)), -0.78, 0.78
    )

    if attack * ball_x > half_len - 0.95:
        vx = clamp(vx + 0.18, 0.45, 1.0)
        vy = clamp(0.42 * vy - 0.42 * ball_y, -0.72, 0.72)
        w = clamp(math.radians(yaw_err) * 0.38, -0.52, 0.52)
        if abs(ball_y) > 1.1:
            vx = min(1.0, vx + 0.06)
            vy = clamp(vy - 0.14 * ball_y, -0.78, 0.78)

    _set_behavior(
        agent,
        "penalty_mode_drive_shot",
        ball_dist=round(ball_dist, 3),
        ball_x=round(ball_x, 2),
        ball_y=round(ball_y, 2),
        local_x=round(b_x, 2),
        local_y=round(b_y, 2),
        cmd_x=round(vx, 2),
        cmd_y=round(vy, 2),
        cmd_w=round(w, 2),
    )
    agent.cmd_vel(vx, vy, w)
    agent.move_head(math.inf, math.inf)


def _penalty_game(agent, tracker):
    gc = agent.gamecontroller
    if gc.penalty != 0:
        _set_behavior(agent, "penalty_mode_penalized_stop", penalty=gc.player_penalty_name)
        agent.stop()
        return

    if gc.game_state == "STATE_INITIAL":
        _set_behavior(agent, "penalty_mode_initial_stop")
        agent.stop()
        return

    if gc.game_state == "STATE_READY":
        tx, ty, tyaw = _penalty_ready_pose(agent, tracker)
        _set_behavior(
            agent, "penalty_mode_ready_pose", target_x=round(tx, 2), target_y=round(ty, 2)
        )
        _drive_to_pose(agent, tx, ty, face_yaw_deg=tyaw, stop_dist=0.12)
        return

    if gc.game_state == "STATE_SET":
        _set_behavior(agent, "penalty_mode_set_stop")
        agent.stop()
        return

    if gc.game_state in ("STATE_FINISHED", "STATE_STANDBY"):
        _set_behavior(agent, "penalty_mode_finished_stop")
        agent.stop()
        return

    if gc.game_state != "STATE_PLAYING":
        _set_behavior(agent, "penalty_mode_unknown_stop", state=gc.game_state)
        agent.stop()
        return

    ball_map = _get_team_ball(agent, tracker)
    if _penalty_boundary_recovery(agent, tracker, ball_map):
        return

    if agent.role == "keeper":
        _keeper_anchor(agent, tracker)
        return

    if agent.role != "striker":
        _penalty_support_hold(agent, ball_map)
        return

    _penalty_drive_shot(agent, tracker, ball_map)


def _baseline_game(agent, tracker):
    gc = agent.gamecontroller
    if gc.penalty != 0:
        agent.stop()
        return

    if gc.game_state == "STATE_INITIAL":
        agent.stop()
        return

    if gc.game_state == "STATE_READY":
        tx, ty, tyaw = _get_ready_pose(agent)
        _drive_to_pose(agent, tx, ty, face_yaw_deg=tyaw, stop_dist=0.18)
        return

    if gc.game_state in ("STATE_SET", "STATE_FINISHED", "STATE_STANDBY"):
        agent.stop()
        return

    if gc.game_state != "STATE_PLAYING":
        agent.stop()
        return

    if not agent.get_if_ball():
        agent.state_machine_runners["find_ball"]()
        return

    if agent.get_ball_distance() > agent.default_chase_distance:
        agent.state_machine_runners["chase_ball"]()
        return

    agent.state_machine_runners["adv_dribble"]()
