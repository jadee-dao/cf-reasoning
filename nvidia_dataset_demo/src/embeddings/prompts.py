# Centralized prompts for VLM strategies

SCENE_DESCRIPTION_PROMPT = (
    "Analyze this scene. List details for:\n"
    "1. Environment (weather, time)\n"
    "2. Ego State (motion, location)\n"
    "3. Key Objects.\n"
    "Format: 'Category: item, item'. No sentences."
)

HAZARD_IDENTIFICATION_PROMPT = (
    "List immediate hazards:\n"
    "1. Collision Risks\n"
    "2. Road Irregularities\n"
    "3. Traffic Control\n"
    "Format: 'Risk: item, item'. Keywords only."
)

STORYBOARD_PROMPT = (
    "Analyze this 2x2 storyboard (time progresses Left->Right, Top->Bottom) as a continuous sequence.\n"
    "Provide a concise summary for:\n"
    "1. Evolution: How traffic and ego state change over time (e.g., 'approaching intersection -> stopped -> turn').\n"
    "2. Developing Hazards: Hazards that appear or persist (e.g., 'pedestrian entering crosswalk', 'vehicle cutting in').\n"
    "3. Overall Difficulty: Low/Medium/High + reason.\n"
    "Do NOT describe individual frames. Synthesize the timeline."
)
