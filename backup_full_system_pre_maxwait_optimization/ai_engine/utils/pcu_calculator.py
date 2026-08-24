"""
Passenger Car Unit (PCU) & Traffic Density Calculator
Provides standard traffic engineering vehicle weighting and queue density estimation.
"""

PCU_FACTORS = {
    "motorcycle": 0.5,
    "car": 1.0,
    "van": 1.2,
    "bus": 2.5,
    "truck": 3.0,
    "ambulance": 1.0  # Given priority flag separately
}

def get_pcu_weight(vehicle_type: str) -> float:
    """Return the PCU weight for a given vehicle type."""
    return PCU_FACTORS.get(vehicle_type.lower(), 1.0)

def calculate_lane_pcu(vehicles: list) -> float:
    """Calculate total PCU load for a list of vehicle objects or vehicle type strings."""
    total_pcu = 0.0
    for v in vehicles:
        if hasattr(v, 'vehicle_type'):
            v_type = v.vehicle_type
        elif isinstance(v, str):
            v_type = v
        elif isinstance(v, dict):
            v_type = v.get('type', 'car')
        else:
            v_type = 'car'
        total_pcu += get_pcu_weight(v_type)
    return round(total_pcu, 2)

def classify_traffic_density(pcu_value: float) -> str:
    """
    Categorize traffic volume based on PCU load:
    - LOW: 0 - 5 PCU
    - MEDIUM: 5 - 12 PCU
    - HIGH: 12 - 20 PCU
    - VERY_HIGH: > 20 PCU
    """
    if pcu_value <= 5.0:
        return "LOW"
    elif pcu_value <= 12.0:
        return "MEDIUM"
    elif pcu_value <= 20.0:
        return "HIGH"
    else:
        return "VERY_HIGH"
