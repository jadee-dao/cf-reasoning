
import os
import glob

def patch_internvl_config():
    # Base path for huggingface modules
    base_path = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/OpenGVLab/InternVideo2_5_Chat_8B")
    
    # Find the specific version directory (hash)
    # matches .../InternVideo2_5_Chat_8B/HASH/configuration_internvl_chat.py
    search_pattern = os.path.join(base_path, "*", "configuration_internvl_chat.py")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No configuration file found in {base_path}")
        return
    
    for file_path in files:
        print(f"Patching {file_path}...")
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False
        for line in lines:
            # Look for the problematic line: output['llm_config'] = self.llm_config.to_dict()
            if "output['llm_config'] = self.llm_config.to_dict()" in line:
                indent = line[:line.find("output")]
                new_line = f"{indent}if hasattr(self, 'llm_config') and self.llm_config is not None:\n{indent}    output['llm_config'] = self.llm_config.to_dict()\n"
                new_lines.append(new_line)
                modified = True
                print("  -> Applied fix for llm_config")
            else:
                new_lines.append(line)
        
        if modified:
            with open(file_path, "w") as f:
                f.writelines(new_lines)
            print("  -> File saved.")
        else:
            print("  -> No changes needed (already patched?).")

if __name__ == "__main__":
    patch_internvl_config()
