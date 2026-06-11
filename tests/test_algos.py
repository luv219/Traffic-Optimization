import unittest
import numpy as np
from controller.algos import FixedTimeController, ActuatedController, AdaptiveController
from controller.simulator import TrafficSimulator

class TestTrafficControllers(unittest.TestCase):
    def test_fixed_time_controller(self):
        controller = FixedTimeController(green_duration=10, yellow_duration=3)
        
        # NS Green phase timer at 5 -> should KEEP
        state = {
            "current_phase": "NS_GREEN",
            "phase_timer": 5,
            "queues": {"N": 2, "S": 2, "E": 0, "W": 0}
        }
        self.assertEqual(controller.decide(state), "KEEP")
        
        # NS Green phase timer at 10 -> should SWITCH (triggers yellow)
        state["phase_timer"] = 10
        self.assertEqual(controller.decide(state), "SWITCH")

        # NS Yellow phase timer at 2 -> should KEEP
        state = {
            "current_phase": "NS_YELLOW",
            "phase_timer": 2,
            "queues": {"N": 2, "S": 2, "E": 0, "W": 0}
        }
        self.assertEqual(controller.decide(state), "KEEP")

        # NS Yellow phase timer at 3 -> should SWITCH
        state["phase_timer"] = 3
        self.assertEqual(controller.decide(state), "SWITCH")

    def test_actuated_controller(self):
        controller = ActuatedController(min_green=10, max_green=40, yellow_duration=3, threshold=1.0)
        
        # NS Green, active traffic, wait queues empty, timer at 15 -> KEEP
        state = {
            "current_phase": "NS_GREEN",
            "phase_timer": 15,
            "queues": {"N": 5, "S": 5, "E": 0, "W": 0}
        }
        self.assertEqual(controller.decide(state), "KEEP")
        
        # NS Green, active lane empty, waiting traffic on EW, timer > min_green -> SWITCH
        state = {
            "current_phase": "NS_GREEN",
            "phase_timer": 15,
            "queues": {"N": 0.5, "S": 0.3, "E": 3.0, "W": 2.0}
        }
        self.assertEqual(controller.decide(state), "SWITCH")

        # NS Green, timer below min_green -> KEEP regardless of traffic
        state["phase_timer"] = 5
        self.assertEqual(controller.decide(state), "KEEP")

        # NS Green, timer exceeds max_green -> SWITCH regardless of active traffic
        state["phase_timer"] = 45
        self.assertEqual(controller.decide(state), "SWITCH")

    def test_adaptive_controller(self):
        controller = AdaptiveController(min_green=10, max_green=40, yellow_duration=3, pressure_threshold=5.0)
        
        # NS Green, NS pressure = 10, EW pressure = 12 (diff 2 < threshold 5) -> KEEP
        state = {
            "current_phase": "NS_GREEN",
            "phase_timer": 15,
            "queues": {"N": 5, "S": 5, "E": 6, "W": 6}
        }
        self.assertEqual(controller.decide(state), "KEEP")

        # NS Green, NS pressure = 4, EW pressure = 12 (diff 8 > threshold 5) -> SWITCH
        state = {
            "current_phase": "NS_GREEN",
            "phase_timer": 15,
            "queues": {"N": 2, "S": 2, "E": 6, "W": 6}
        }
        self.assertEqual(controller.decide(state), "SWITCH")

        # NS Green, NS pressure = 4, EW pressure = 12, but timer < min_green -> KEEP
        state["phase_timer"] = 5
        self.assertEqual(controller.decide(state), "KEEP")


class TestTrafficSimulator(unittest.TestCase):
    def test_simulator_basic_stepping(self):
        sim = TrafficSimulator(departure_rate=1.0)
        sim.reset()
        
        # Verify initial conditions
        self.assertEqual(sim.time_elapsed, 0)
        self.assertEqual(sim.current_phase, "NS_GREEN")
        self.assertEqual(sum(sim.queues.values()), 0.0)
        
        # Set deterministic arrival rates (no random arrivals for testing)
        sim.set_arrival_rates({"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0})
        
        # Step once
        sim.step()
        self.assertEqual(sim.time_elapsed, 1)
        self.assertEqual(sim.phase_timer, 1)
        
        # Inject some queues manually
        sim.queues = {"N": 5.0, "S": 3.0, "E": 2.0, "W": 2.0}
        
        # Step under NS_GREEN. Departure rate is 1.0.
        # N and S should decrease by 1.0, E and W should remain 2.0.
        sim.step()
        self.assertEqual(sim.queues["N"], 4.0)
        self.assertEqual(sim.queues["S"], 2.0)
        self.assertEqual(sim.queues["E"], 2.0)
        self.assertEqual(sim.queues["W"], 2.0)
        self.assertEqual(sim.throughput, 2.0)

if __name__ == "__main__":
    unittest.main()
