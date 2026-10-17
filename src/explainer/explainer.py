import json
import time
import os

def build_granite_prompt(scenario_name, input_weights, opt_data):
    plan = opt_data.get("plan", {})
    saved_co2e = opt_data.get("saved_kgco2e", 0.0)
    notes = opt_data.get("notes", [])
    
    prompt = f"""<|system|>
You are an AI sustainability advisor for PolyRoute. Explain the recycling batch results to a non-technical user.
Rules:
1. Output exactly 3 to 5 sentences.
2. State the total CO2e saved.
3. Explain the routing decisions in simple, everyday language (e.g., mentioning facility space or material recyclability). 
4. DO NOT mention algorithms, databases, 'WARM factors', or how the math works behind the scenes.
5. If a plastic had to be sent to a landfill or combusted, point out why (e.g., the recycling facility was full, or that specific plastic isn't recyclable).
6. DO NOT invent or calculate any numbers. Use only the numbers provided.

<|user|>
Batch Context: {scenario_name}
Input Weights (kg): {json.dumps(input_weights)}

Optimization Results:
- Total CO2e Saved: {saved_co2e} kg
- Optimized Routing Plan: {json.dumps(plan)}
- Engine Notes: {', '.join(notes) if notes else 'None'}

<|assistant|>
"""
    return prompt
def generate_fallback(saved_co2e, plan, notes):
    fallback_text = f"This batch optimization saved {saved_co2e} kg of CO2e compared to baseline disposal. "
    
    for polymer, routes in plan.items():
        for route, amount in routes.items():
            fallback_text += f"Routed {amount} kg of {polymer} to {route}. "
            
    if notes:
        fallback_text += "Notes: " + " ".join(notes)
        
    return fallback_text.strip()

def generate_explanation(prompt, opt_data, use_local_model=True, model_path="./granite-4.0-h-instruct.gguf"):
    start_time = time.time()
    
    plan = opt_data.get("plan", {})
    saved_co2e = opt_data.get("saved_kgco2e", 0.0)
    notes = opt_data.get("notes", [])
    
    if not use_local_model:
        return generate_fallback(saved_co2e, plan, notes), 0.0

    try:
        from llama_cpp import Llama
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model {model_path} not found.")
            
        llm = Llama(model_path=model_path,n_gpu_layers=-1, n_ctx=1024, verbose=False)
        
        output = llm(
            prompt,
            max_tokens=250,
            temperature=0.1,
            stop=["<|user|>", "<|system|>"]
        )
        
        response_text = output['choices'][0]['text'].strip()
        elapsed_time = round(time.time() - start_time, 2)
        
        return response_text, elapsed_time

    except ImportError:
        print("[LLM Warning] llama-cpp-python not installed. Using template fallback.")
        elapsed_time = round(time.time() - start_time, 2)
        return generate_fallback(saved_co2e, plan, notes), elapsed_time
        
    except Exception as e:
        print(f"[LLM Warning] Local model failed ({str(e)}). Using template fallback.")
        elapsed_time = round(time.time() - start_time, 2)
        return generate_fallback(saved_co2e, plan, notes), elapsed_time

if __name__ == "__main__":
    input_file = "src/optimiser/optimized_routes_output.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run route_opt.py first.")
    else:
        with open(input_file, "r") as f:
            optimizations = json.load(f)
            
        print("=== PolyRoute LLM Explainer (Local Granite 4.0) ===\n")
        
        for case in optimizations[:2]:
            print(f"--- Scenario: {case['scenario_name']} ---")
            
            prompt = build_granite_prompt(
                case["scenario_name"], 
                case["input_weights_kg"], 
                case["output"]
            )
            
            # The function will automatically fall back if the model file isn't present
            explanation, latency = generate_explanation(
                prompt, 
                case["output"], 
                use_local_model=True 
            )
            
            print(f"Explanation (Generation time: {latency}s):")
            print(f"> {explanation}\n")
