import pulp

def optimize_routes(composition_data, facility_limits, warm_factors):
    # Early exit if the composition data is empty
    if not composition_data or all(weight == 0 for weight in composition_data.values()):
        return {
            "plan": {},
            "optimised_kgco2e": 0.0,
            "baseline_kgco2e": 0.0,
            "saved_kgco2e": 0.0,
            "notes": ["Empty or zero-weight batch received."]
        }

    prob = pulp.LpProblem("PolyRoute_Carbon_Minimization", pulp.LpMinimize)
    
    # Decision Variables: x[polymer, route] >= 0
    # Filter out any polymers passed in composition_data that aren't in warm_factors
    polymers = [p for p in composition_data.keys() if p in warm_factors]
    routes = ["recycle", "compost", "landfill", "combust"]
    
    x = {}
    for p in polymers:
        for r in routes:
            if warm_factors[p].get(r) != "NA":
                x[(p, r)] = prob.add_variable(f"route_{p}_{r}", lowBound=0, cat='Continuous')

    # Objective Function: Minimize total CO2e
    prob += pulp.lpSum([x[(p, r)] * warm_factors[p][r] for (p, r) in x.keys()])
    
    # Constraints
    # 1) Each valid polymer must be fully routed
    for p in polymers:
        prob += pulp.lpSum([x[(p, r)] for r in routes if (p, r) in x.keys()]) == composition_data[p]
        
    # 2) Facility Capacity Limits
    prob += pulp.lpSum([x[(p, "recycle")] for p in polymers if (p, "recycle") in x.keys()]) <= facility_limits.get("recycle_cap_kg", 0)
    prob += pulp.lpSum([x[(p, "compost")] for p in polymers if (p, "compost") in x.keys()]) <= facility_limits.get("compost_cap_kg", 0)
    prob += pulp.lpSum([x[(p, "landfill")] for p in polymers if (p, "landfill") in x.keys()]) <= facility_limits.get("landfill_cap_kg", 0)
    
    status = prob.solve(pulp.GLPK_CMD(msg=False))
    if "Optimal" not in str(status):
        return {"error": "Infeasible: Facility limits are too tight for the batch weight."}
        
    plan = {}
    notes = []
    
    for p in composition_data.keys():
        if p not in warm_factors:
            notes.append(f"{p}: unknown polymer type, skipped by optimiser.")
            continue
            
        polymer_plan = {}
        for r in routes:
            if (p, r) in x.keys():
                val = x[(p, r)].varValue
                if val is not None and val > 0.0:
                    polymer_plan[r] = round(val, 2)
        
        if polymer_plan:
            plan[p] = polymer_plan
            
        if warm_factors[p].get("recycle") == "NA":
            notes.append(f"{p}: no recycling factor in WARM, routed to other facilities.")
            
    # Calculate objective value (Optimised CO2e)
    optimised_kgco2e = round(pulp.value(prob.objective), 2)
    
    # Calculate baseline CO2e (Assumption: Current practice sends everything to landfill/combust)
    baseline_kgco2e = 0.0
    for p, weight in composition_data.items():
        if p in warm_factors:
            if warm_factors[p].get("landfill") != "NA":
                baseline_kgco2e += weight * warm_factors[p]["landfill"]
            elif warm_factors[p].get("combust") != "NA":
                baseline_kgco2e += weight * warm_factors[p]["combust"]
            
    baseline_kgco2e = round(baseline_kgco2e, 2)
    saved_kgco2e = round(baseline_kgco2e - optimised_kgco2e, 2)
    
    return {
        "plan": plan,
        "optimised_kgco2e": optimised_kgco2e,
        "baseline_kgco2e": baseline_kgco2e,
        "saved_kgco2e": saved_kgco2e,
        "notes": notes
    }

if __name__ == "__main__":
    import json
    import os

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

    json_file = "generated_compositions.json"
    
    if not os.path.exists(json_file):
        print(f"File '{json_file}' not found. Please run the generator script first.")
    else:
        with open(json_file, "r") as f:
            scenarios = json.load(f)

        print("=== PolyRoute Route Optimiser Test ===")
        print(f"Facility Limits (kg): {test_facility_limits}")
        
        for case in scenarios:
            print(f"\n--- Testing Scenario {case['scenario_id']}: {case['scenario_name']} ---")
            
            comp_kg = case["composition_data"]["kg"]
            print(f"Input Weights (kg): {comp_kg}")
            
            result = optimize_routes(comp_kg, test_facility_limits, test_warm_factors)
            
            if "error" in result:
                print(f"OPTIMISATION FAILED: {result['error']}")
            else:
                print(json.dumps(result, indent=4))
