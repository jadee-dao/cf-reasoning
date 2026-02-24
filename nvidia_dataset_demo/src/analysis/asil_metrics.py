import re
import json

def calculate_asil(severity, exposure, controllability):
    """
    Calculates the ASIL level based on ISO 26262 tables.
    
    Args:
        severity (int): S0-S3 (0-3)
        exposure (int): E0-E4 (0-4)
        controllability (int): C0-C3 (0-3)
        
    Returns:
        str: ASIL Level (QM, A, B, C, D)
    """
    # Simply mapping: S + E + C = Score? No, it's a look-up table.
    # Simplified Logic based on standard tables:
    # QM is default.
    # D is highest risk.
    
    if severity == 0 or exposure == 0 or controllability == 0:
        return "QM"
        
    # Table logic (approximation for standard ISO 26262 Part 3)
    # S1, S2, S3
    # E1, E2, E3, E4
    # C1, C2, C3
    
    # Map to simpler integer sum for heuristic? 
    # Let's use explicit mapping for correctness.
    
    table = {
        # S1
        (1, 1, 1): "QM", (1, 1, 2): "QM", (1, 1, 3): "QM",
        (1, 2, 1): "QM", (1, 2, 2): "QM", (1, 2, 3): "QM",
        (1, 3, 1): "QM", (1, 3, 2): "QM", (1, 3, 3): "A",
        (1, 4, 1): "QM", (1, 4, 2): "A",  (1, 4, 3): "B",
        
        # S2
        (2, 1, 1): "QM", (2, 1, 2): "QM", (2, 1, 3): "QM",
        (2, 2, 1): "QM", (2, 2, 2): "QM", (2, 2, 3): "A",
        (2, 3, 1): "QM", (2, 3, 2): "A",  (2, 3, 3): "B",
        (2, 4, 1): "A",  (2, 4, 2): "B",  (2, 4, 3): "C",
        
        # S3
        (3, 1, 1): "QM", (3, 1, 2): "QM", (3, 1, 3): "A",
        (3, 2, 1): "QM", (3, 2, 2): "A",  (3, 2, 3): "B",
        (3, 3, 1): "A",  (3, 3, 2): "B",  (3, 3, 3): "C",
        (3, 4, 1): "B",  (3, 4, 2): "C",  (3, 4, 3): "D",
    }
    
    key = (severity, exposure, controllability)
    return table.get(key, "QM")

def asil_to_score(asil_level):
    """Converts ASIL level to a numeric score (0-4)."""
    mapping = {"QM": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    return mapping.get(asil_level, 0)

def parse_llm_response(text):
    """
    Parses the JSON part of the LLM response.
    Expected format: ```json { ... } ``` or just { ... }
    
    Returns:
        dict: Parsed metrics or None if failed.
    """
    try:
        # Try to find JSON block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
            else:
                return None
        
        data = json.loads(json_str)
        
        # Validate keys
        required = ["severity", "exposure", "controllability", "reasoning"]
        for k in required:
            if k not in data:
                print(f"Missing key {k} in LLM response.")
                return None
                
        # Ensure values are ints
        data["severity"] = int(data["severity"])
        data["exposure"] = int(data["exposure"])
        data["controllability"] = int(data["controllability"])
        
        # Recalculate ASIL to be safe
        data["asil_level"] = calculate_asil(data["severity"], data["exposure"], data["controllability"])
        data["asil_score"] = asil_to_score(data["asil_level"])
        
        return data
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None
