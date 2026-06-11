import random
import time
from controller.algos import FixedTimeController, ActuatedController, AdaptiveController

class TrafficSimulator:
    def __init__(self, departure_rate=0.8, yellow_duration=3):
        # Queues in lanes: North (N), South (S), East (E), West (W)
        self.queues = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
        
        # Arrival rates (vehicles per second) for each lane
        self.arrival_rates = {"N": 0.15, "S": 0.15, "E": 0.10, "W": 0.10}
        
        # Departure rate (vehicles per second from green lanes)
        self.departure_rate = departure_rate
        self.yellow_duration = yellow_duration
        
        # Simulation state
        self.current_phase = "NS_GREEN"  # NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW
        self.phase_timer = 0
        self.time_elapsed = 0
        
        # Performance metrics
        self.throughput = 0.0
        self.total_delay = 0.0
        self.history = []  # list of dicts holding step stats
        
        # Controller
        self.controller_type = "Fixed-Time"
        self.controller = FixedTimeController(yellow_duration=self.yellow_duration)
        
    def reset(self):
        self.queues = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
        self.phase_timer = 0
        self.time_elapsed = 0
        self.throughput = 0.0
        self.total_delay = 0.0
        self.history = []
        self.set_controller(self.controller_type)

    def set_arrival_rates(self, rates):
        """Rates dict, e.g. {'N': 0.2, 'S': 0.2, 'E': 0.1, 'W': 0.1}"""
        self.arrival_rates.update(rates)

    def set_controller(self, name):
        self.controller_type = name
        if name == "Fixed-Time":
            self.controller = FixedTimeController(yellow_duration=self.yellow_duration)
        elif name == "Actuated":
            self.controller = ActuatedController(yellow_duration=self.yellow_duration)
        elif name == "Adaptive":
            self.controller = AdaptiveController(yellow_duration=self.yellow_duration)
        else:
            raise ValueError(f"Unknown controller type: {name}")

    def step(self):
        """Advances the simulation by 1 second."""
        self.time_elapsed += 1
        self.phase_timer += 1
        
        # 1. Simulate vehicle arrivals (Stochastic arrival)
        for lane, rate in self.arrival_rates.items():
            # Random arrival based on rate (approximate Poisson process)
            if random.random() < rate:
                self.queues[lane] += 1.0
                
        # 2. Simulate departures from green lanes
        green_lanes = []
        if self.current_phase == "NS_GREEN":
            green_lanes = ["N", "S"]
        elif self.current_phase == "EW_GREEN":
            green_lanes = ["E", "W"]
            
        for lane in green_lanes:
            if self.queues[lane] > 0:
                # Departures can be fractional/continuous or discrete. Let's make it continuous:
                departed = min(self.queues[lane], self.departure_rate)
                self.queues[lane] -= departed
                self.throughput += departed
                
        # 3. Calculate delay (queues wait for 1 second)
        current_queue_sum = sum(self.queues.values())
        self.total_delay += current_queue_sum
        
        # 4. Get controller decision
        state = {
            "current_phase": self.current_phase,
            "phase_timer": self.phase_timer,
            "queues": self.queues.copy()
        }
        
        decision = self.controller.decide(state)
        
        # 5. Handle phase transitions
        if decision == "SWITCH":
            self.phase_timer = 0
            if self.current_phase == "NS_GREEN":
                self.current_phase = "NS_YELLOW"
            elif self.current_phase == "NS_YELLOW":
                self.current_phase = "EW_GREEN"
            elif self.current_phase == "EW_GREEN":
                self.current_phase = "EW_YELLOW"
            elif self.current_phase == "EW_YELLOW":
                self.current_phase = "NS_GREEN"
                
        # 6. Record history (limit size to avoid memory growth)
        history_item = {
            "time": self.time_elapsed,
            "queues": self.queues.copy(),
            "phase": self.current_phase,
            "throughput": self.throughput,
            "total_delay": self.total_delay,
            "avg_delay": self.total_delay / max(1.0, self.throughput)
        }
        self.history.append(history_item)
        if len(self.history) > 300:  # Keep last 5 minutes
            self.history.pop(0)
            
        return history_item

    def get_status(self):
        return {
            "time_elapsed": self.time_elapsed,
            "current_phase": self.current_phase,
            "phase_timer": self.phase_timer,
            "queues": self.queues.copy(),
            "throughput": round(self.throughput, 1),
            "total_delay": round(self.total_delay, 1),
            "avg_delay": round(self.total_delay / max(1.0, self.throughput), 2),
            "controller_type": self.controller_type,
            "arrival_rates": self.arrival_rates.copy()
        }

if __name__ == "__main__":
    # Small test
    sim = TrafficSimulator()
    print("Testing Simulator with Fixed-Time Controller:")
    for _ in range(35):
        sim.step()
    status = sim.get_status()
    print("Time elapsed:", status["time_elapsed"])
    print("Current phase:", status["current_phase"])
    print("Queues:", status["queues"])
    print("Throughput:", status["throughput"])
    print("Avg Delay per vehicle:", status["avg_delay"])
