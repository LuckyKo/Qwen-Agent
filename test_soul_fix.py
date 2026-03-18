import sys
from pathlib import Path
sys.path.append('.')
from soul_loader import load_soul, build_system_prompt

def test():
    soul_path = 'agents/orchestrator_soul.md'
    print(f"Testing with: {soul_path}")
    
    try:
        config = load_soul(soul_path)
        prompt = build_system_prompt(config)
        
        print("\n--- GENERATED PROMPT PREVIEW ---\n")
        print(prompt)
        print("\n--- END PREVIEW ---\n")
        
        # Check for rules formatting
        if "1. DELEGATE FIRST" in prompt and "{" not in prompt.split("## Your Rules")[1].split("##")[0]:
            print("[PASS] Rules are correctly formatted (no brackets).")
        else:
            print("[FAIL] Rules formatting issue detected.")
            
        # Check for dynamic sections
        expected_sections = ["## Core Responsibilities", "## Delegation Guidelines", "## Operation Workflow"]
        for section in expected_sections:
            if section in prompt:
                print(f"[PASS] Section found: {section}")
            else:
                print(f"[FAIL] Missing section: {section}")
                
        # Check for remember section
        if "## Remember" in prompt:
            print("[PASS] Remember section found.")
        else:
            print("[FAIL] Missing Remember section.")
            
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    test()
