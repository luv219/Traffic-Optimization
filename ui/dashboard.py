import os
import json
import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

from controller.simulator import TrafficSimulator
from counting.counter import VehicleCounter

# Initialize FastAPI
app = FastAPI(title="Adaptive Traffic Optimization Dashboard")

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_DIR = os.path.join(ROOT_DIR, "ui", "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global Simulation & Counter instances
simulator = TrafficSimulator()
vehicle_counter = None

# Simulation configuration variables
sim_speed = 1.0  # multiplier for simulator sleep time (1.0 = 1s delay)
sim_running = True

class ConfigUpdate(BaseModel):
    controller_type: str
    arrival_rates: dict
    sim_speed: float

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/video_feed")
def video_feed():
    """Endpoint for streaming YOLO-processed traffic video feed."""
    global vehicle_counter
    if vehicle_counter is None:
        try:
            vehicle_counter = VehicleCounter(
                model_path=os.path.join(ROOT_DIR, "models", "yolo11n.pt"),
                zones_config_path=os.path.join(ROOT_DIR, "counting", "zones.yaml")
            )
        except Exception as e:
            print(f"Error initializing vehicle counter: {e}")
            return StreamingResponse(iter([]))
            
    video_path = os.path.join(ROOT_DIR, "data", "videos", "traffic.mp4")
    if not os.path.exists(video_path):
        print(f"Video file not found at: {video_path}")
        return StreamingResponse(iter([]))
        
    return StreamingResponse(
        vehicle_counter.generate_video_stream(video_path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/live_counts")
def live_counts():
    """Gets real-time object tracking vehicle counts from the video feed."""
    global vehicle_counter
    if vehicle_counter is not None:
        return vehicle_counter.get_counts()
    return {"current": {}, "cumulative": {}}

@app.get("/simulation_stream")
async def simulation_stream(request: Request):
    """Server-Sent Events (SSE) stream for real-time simulator metrics."""
    async def event_generator():
        global sim_running, sim_speed
        while True:
            # If client disconnects, stop streaming
            if await request.is_disconnected():
                break
                
            if sim_running:
                # Step the simulator
                simulator.step()
                
            # Get current simulator state
            status = simulator.get_status()
            
            # Incorporate live video counts if the video feed is active
            global vehicle_counter
            if vehicle_counter is not None:
                video_counts = vehicle_counter.get_counts()
                status["video_counts"] = video_counts
            else:
                status["video_counts"] = {"current": {}, "cumulative": {}}
                
            yield f"data: {json.dumps(status)}\n\n"
            
            # Control simulation speed
            sleep_time = max(0.1, 1.0 / sim_speed)
            await asyncio.sleep(sleep_time)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/config")
async def update_config(data: ConfigUpdate):
    """Updates simulator controller type, arrival rates, and speed."""
    global sim_speed
    try:
        simulator.set_controller(data.controller_type)
        simulator.set_arrival_rates(data.arrival_rates)
        sim_speed = data.sim_speed
        return {"status": "success", "config": simulator.get_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/control/pause")
def pause_sim():
    global sim_running
    sim_running = False
    return {"status": "paused"}

@app.post("/control/resume")
def resume_sim():
    global sim_running
    sim_running = True
    return {"status": "running"}

@app.post("/control/reset")
def reset_sim():
    simulator.reset()
    return {"status": "reset"}

@app.get("/compare")
def compare_controllers():
    """
    Runs a parallel 200-step simulation of all three controllers
    under the same traffic arrival rates, and returns compared performance metrics.
    """
    rates = simulator.arrival_rates.copy()
    results = {}
    
    for algo in ["Fixed-Time", "Actuated", "Adaptive"]:
        # Run test sim
        test_sim = TrafficSimulator(departure_rate=simulator.departure_rate, yellow_duration=simulator.yellow_duration)
        test_sim.set_arrival_rates(rates)
        test_sim.set_controller(algo)
        
        # Collect queue lengths, delays, and throughput
        queue_lengths = []
        for _ in range(200):
            test_sim.step()
            queue_lengths.append(sum(test_sim.queues.values()))
            
        status = test_sim.get_status()
        results[algo] = {
            "throughput": status["throughput"],
            "total_delay": status["total_delay"],
            "avg_delay": status["avg_delay"],
            "avg_queue": round(sum(queue_lengths) / len(queue_lengths), 2)
        }
        
    return results

if __name__ == "__main__":
    print("Starting FastAPI Dashboard server...")
    uvicorn.run("ui.dashboard:app", host="127.0.0.1", port=8000, reload=True)
