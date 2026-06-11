import numpy as np

class BaseController:
    def __init__(self, yellow_duration=3, min_green=10, max_green=60):
        self.yellow_duration = yellow_duration
        self.min_green = min_green
        self.max_green = max_green

    def decide(self, state):
        """
        Given the intersection state, returns the desired control action.
        Returns:
            action: "KEEP" or "SWITCH"
        """
        raise NotImplementedError

class FixedTimeController(BaseController):
    def __init__(self, green_duration=30, yellow_duration=3):
        super().__init__(yellow_duration)
        self.green_duration = green_duration

    def decide(self, state):
        # state is a dict containing:
        # "current_phase": str ('NS_GREEN', 'NS_YELLOW', 'EW_GREEN', 'EW_YELLOW')
        # "phase_timer": int (seconds elapsed in current phase)
        # "queues": dict {'N': val, 'S': val, 'E': val, 'W': val}
        phase = state["current_phase"]
        timer = state["phase_timer"]
        
        if phase in ["NS_GREEN", "EW_GREEN"]:
            if timer >= self.green_duration:
                return "SWITCH"
        elif phase in ["NS_YELLOW", "EW_YELLOW"]:
            if timer >= self.yellow_duration:
                return "SWITCH"
        return "KEEP"

class ActuatedController(BaseController):
    def __init__(self, min_green=10, max_green=60, yellow_duration=3, extension_time=5, threshold=1.0):
        super().__init__(yellow_duration, min_green, max_green)
        self.extension_time = extension_time
        self.threshold = threshold  # Queue threshold to consider "traffic present"

    def decide(self, state):
        phase = state["current_phase"]
        timer = state["phase_timer"]
        queues = state["queues"]
        
        if phase in ["NS_YELLOW", "EW_YELLOW"]:
            if timer >= self.yellow_duration:
                return "SWITCH"
            return "KEEP"
            
        # For Green phases
        if timer < self.min_green:
            return "KEEP"
            
        if timer >= self.max_green:
            return "SWITCH"
            
        # Get active and waiting queues
        if phase == "NS_GREEN":
            active_q = queues["N"] + queues["S"]
            waiting_q = queues["E"] + queues["W"]
        else:
            active_q = queues["E"] + queues["W"]
            waiting_q = queues["N"] + queues["S"]
            
        # Actuated extension logic:
        # If there are vehicles waiting on red lane and active lane is relatively empty, switch
        if waiting_q > self.threshold and active_q <= self.threshold:
            return "SWITCH"
            
        return "KEEP"

class AdaptiveController(BaseController):
    """
    Max-Pressure inspired adaptive traffic light controller.
    It switches green lights dynamically to the axis with higher traffic queue pressure.
    """
    def __init__(self, min_green=10, max_green=60, yellow_duration=3, pressure_threshold=5.0):
        super().__init__(yellow_duration, min_green, max_green)
        self.pressure_threshold = pressure_threshold  # Pressure difference required to switch

    def decide(self, state):
        phase = state["current_phase"]
        timer = state["phase_timer"]
        queues = state["queues"]
        
        if phase in ["NS_YELLOW", "EW_YELLOW"]:
            if timer >= self.yellow_duration:
                return "SWITCH"
            return "KEEP"
            
        if timer < self.min_green:
            return "KEEP"
            
        if timer >= self.max_green:
            return "SWITCH"
            
        # Calculate pressures (sum of queues for the phase lanes)
        ns_pressure = queues["N"] + queues["S"]
        ew_pressure = queues["E"] + queues["W"]
        
        # Decide based on pressure difference
        if phase == "NS_GREEN":
            # If EW pressure is significantly higher than NS, switch
            if ew_pressure > ns_pressure + self.pressure_threshold:
                return "SWITCH"
        elif phase == "EW_GREEN":
            # If NS pressure is significantly higher than EW, switch
            if ns_pressure > ew_pressure + self.pressure_threshold:
                return "SWITCH"
                
        return "KEEP"
