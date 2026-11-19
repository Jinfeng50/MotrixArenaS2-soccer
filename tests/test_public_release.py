"""Regression checks that do not require the physics simulator."""

import importlib
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decider"))

import configuration
import user_entry
from interfaces.sim_client import SimClient


class PublicReleaseTests(unittest.TestCase):
    def test_configuration_and_legacy_manager_import(self):
        self.assertEqual(configuration.load_config()["league"], "S")
        module = importlib.import_module("strategy.team_manager")
        self.assertTrue(callable(module.AttackStateMachine))

    def test_own_corner_clearance(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                agent = Mock(
                    field_length=9.0,
                    field_width=6.0,
                    local_id=0,
                    role="striker",
                    last_body_push_attempt_ts=0.0,
                )
                agent.get_self_pos.return_value = [-4.2, side * 2.6]
                agent.get_self_yaw.return_value = 0.0
                agent.get_config.return_value = {"id": 0}
                with patch.object(user_entry, "_set_behavior") as behavior:
                    handled = user_entry._body_push_to_goal(
                        agent, Mock(), [-3.8, side * 2.6], 0.4, 0.0, 0.4
                    )
                self.assertTrue(handled)
                self.assertEqual(behavior.call_args.args[1], "body_push_own_endline_corner_escape")
                vx, vy, w = agent.cmd_vel.call_args.args
                self.assertTrue(all(math.isfinite(v) and abs(v) <= 1 for v in (vx, vy, w)))
                self.assertGreater(vx, 0)
                self.assertLess(vy * side, 0)

    def test_bundled_zmq_single_and_batch_protocol(self):
        context = zmq.Context()
        server = context.socket(zmq.REP)
        server.setsockopt(zmq.RCVTIMEO, 2000)
        port = server.bind_to_random_port("tcp://127.0.0.1")
        client = SimClient(port=port)
        client.client.socket.setsockopt(zmq.RCVTIMEO, 2000)
        response = {"state": {"robots": [], "ball": {}}, "sim_timestamp": 1.0}
        original_recv = client.client.recv_json
        requests = []

        def respond():
            requests.append(server.recv_json())
            server.send_json(response)
            return original_recv()

        try:
            with patch.object(client.client, "recv_json", side_effect=respond):
                self.assertEqual(client.communicate([0.2, 0.0, 0.1], robot_id=7), response)
                commands = [{"id": 0, "cmd": [0.0, 0.0, 0.0]}]
                self.assertEqual(client.communicate_batch(commands), response)
            self.assertEqual(requests[0]["id"], 7)
            self.assertEqual(requests[0]["cmd"], [0.2, 0.0, 0.1])
            self.assertEqual(requests[1]["commands"], commands)
            self.assertNotIn("kick", requests[0])
        finally:
            client.close()
            server.close(linger=0)
            context.term()


if __name__ == "__main__":
    unittest.main()
