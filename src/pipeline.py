import json
import os
import time
from optimiser.route_opt import optimize_routes
from explainer.explainer import build_granite_prompt, generate_explanation

def process_batch(scenario_name, composition_kg, limits, factors, use_llm=True):
    """
    The main backend function that your dashboard will call.
    It takes raw plastic weights, optimizes them, and generates an AI explanation.
    """
    print(f"--- Processing: {scenario_name} ---")
    
    # 1. Run the PuLP Optimizer
    print("Running optimization...")
    opt_data = optimize_routes(composition_kg, limits, factors)
    
    # Check if the optimizer threw a capacity error
    if "error" in opt_data:
        return {"status": "failed", "error_message": opt_data["error"]}
        
    # 2. Build the LLM Prompt
    print("Building context for Granite...")
    prompt = build_granite_prompt(scenario_name, composition_kg, opt_data)
    
    # 3. Generate the Explanation
    print("Generating AI explanation (this may take a few seconds)...")
    explanation, latency = generate_explanation(prompt, opt_data, use_local_model=use_llm)
    
    # 4. Return the complete package for the frontend
    return {
        "status": "success",
        "saved_kgco2e": opt_data["saved_kgco2e"],
        "plan": opt_data["plan"],
        "explanation": explanation,
        "llm_latency_seconds": latency
    }

if __name__ == "__main__":
    # Standard WARM factors & Limits matching your lab specs
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
    
    # Path to the generator script's output
    input_json = "src/optimiser/generated_compositions.json"
    
    if not os.path.exists(input_json):
        print(f"Error: '{input_json}' not found. Please run your generator script first.")
    else:
        with open(input_json, "r") as f:
            scenarios = json.load(f)
            
        print(f"Loaded {len(scenarios)} scenarios from {input_json}\n")
        
        # We'll process the first 2 scenarios as a test so you aren't waiting on the GPU for all 10
        for case in scenarios:
            scenario_name = case.get("scenario_name", "Unknown Scenario")
            comp_kg = case.get("composition_data", {}).get("kg", {})
            
            # Run the full pipeline!
            result = process_batch(
                scenario_name=scenario_name,
                composition_kg=comp_kg,
                limits=test_facility_limits,
                factors=test_warm_factors,
                use_llm=True
            )
            
            # Print the final API response
            if result["status"] == "success":
                print("=== FINAL PIPELINE OUTPUT ===")
                print(f"Total CO2e Saved: {result['saved_kgco2e']} kg")
                print(f"Routing Plan: {result['plan']}")
                print(f"AI Summary ({result['llm_latency_seconds']}s): {result['explanation']}\n")
            else:
                print(f"Pipeline Error: {result['error_message']}\n")
