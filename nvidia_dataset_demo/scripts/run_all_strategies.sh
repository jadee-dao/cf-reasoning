#!/bin/bash
set -e

# List of all strategies defined in scripts/run_embedding_test.py
strategies=(
    "naive"
    "foreground_strict"
    "foreground_loose"
    "text"
    "vlm"
    "video"
    "object_semantics"
    "fastvit_attention"
    "fastvlm_description"
    "fastvlm_hazard"
    "openrouter_description"
    "openrouter_hazard"
    "openrouter_storyboard"
)

echo "Starting execution of all strategies..."

for strategy in "${strategies[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running Strategy: $strategy"
    echo "----------------------------------------------------------------"
    
    # Run the python script
    # Allow it to fail without stopping the whole loop? 
    # The user might prefer 'set -e' to stop on error, or continue. 
    # I'll use '|| true' to let it continue if one fails (e.g. missing API key).
    python scripts/run_embedding_test.py --strategy "$strategy" --limit 100 --interval 1.5 || echo "Strategy $strategy failed!"
    
    echo ""
done

echo "Done running all strategies."
