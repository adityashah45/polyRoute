import json
import os
from comp_analyzer.config import SYNTHETIC_DIR
from comp_analyzer.example_for_optimiser import composition_for_optimiser
from optimiser.route_opt import optimize_routes
from explainer.explainer import build_granite_prompt, generate_explanation

def process_batch(scenario_name, composition_data, limits, factors, use_llm=True):
    print(f"\n--- Processing: {scenario_name} ---")
    
    # 1. Run the PuLP Optimizer
    print("Running optimization...")
    opt_data = optimize_routes(composition_data, limits, factors)
    
    if "error" in opt_data:
        return {"status": "failed", "error_message": opt_data["error"]}
        
    # 2. Build the LLM Prompt
    print("Building context for Granite...")
    prompt = build_granite_prompt(scenario_name, composition_data, opt_data)
    
    # 3. Generate the Explanation
    print("Generating AI explanation (this may take a few seconds)...")
    explanation, latency = generate_explanation(prompt, opt_data, use_local_model=use_llm)
    
    return {
        "status": "success",
        "saved_kgco2e": opt_data["saved_kgco2e"],
        "plan": opt_data["plan"],
        "explanation": explanation,
        "llm_latency_seconds": latency
    }

if __name__ == "__main__":
    # Standard WARM factors & Limits
    test_facility_limits = {
        "recycle_cap_kg": 40.0,
        "compost_cap_kg": 10.0,
        "landfill_cap_kg": 15.0
    }

    test_warm_factors = {
        "PET":   {"recycle": -1.2, "compost": "NA", "landfill": 0.1, "combust": 1.5},
        "HDPE":  {"recycle": -1.0, "compost": "NA", "landfill": 0.1, "combust": 1.3},
        "PP":    {"recycle": -0.8, "compost": "NA", "landfill": 0.1, "combust": 1.4},
        "PVC":   {"recycle": "NA",  "compost": "NA", "landfill": 0.2, "combust": 2.0},
        "LDPE":  {"recycle": -0.9, "compost": "NA", "landfill": 0.1, "combust": 1.4},
        "PS":    {"recycle": -1.1, "compost": "NA", "landfill": 0.1, "combust": 1.6},
        "Other": {"recycle": "NA",  "compost": "NA", "landfill": 0.1, "combust": 1.5}
    }
    
    # Test files defined in the integration specs
    test_files = ["medium_100.json", "pet_heavy_100.json", "empty.json"]
    
    for file_name in test_files:
        file_path = SYNTHETIC_DIR / file_name
        
        if not file_path.exists():
            print(f"Skipping {file_name}: File not found in {SYNTHETIC_DIR}")
            continue
            
        with open(file_path, "r") as f:
            items = json.load(f)
            
        # Convert raw classifier items into composition data (assuming 50kg batch)
        composition_data, _ = composition_for_optimiser(items, total_kg=50)
        
        result = process_batch(
            scenario_name=file_name,
            composition_data=composition_data,
            limits=test_facility_limits,
            factors=test_warm_factors,
            use_llm=True
        )
        
        if result["status"] == "success":
            print("=== FINAL PIPELINE OUTPUT ===")
            print(f"Total CO2e Saved: {result['saved_kgco2e']} kg")
            print(f"Routing Plan: {result['plan']}")
            print(f"AI Summary ({result['llm_latency_seconds']}s): {result['explanation']}")
        else:
            print(f"Pipeline Error: {result['error_message']}")
