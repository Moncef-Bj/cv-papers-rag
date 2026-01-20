import json

with open('data/figure_descriptions.json', 'r') as f:
    figures = json.load(f)

print("="*60)
print("CHECKING VISAPP AND ALGORITHM FIGURES")
print("="*60)

found = False
for fig in figures:
    filename = fig.get('filename', '').upper()
    if 'VISAPP' in filename or 'ALGORITHM' in filename:
        found = True
        desc = fig.get('description', '')
        print(f"\nFilename: {fig.get('filename')}")
        print(f"Source: {fig.get('source_pdf')}")
        print(f"Description length: {len(desc)} chars")
        print(f"Description: {desc[:200]}...")
        
        # Check why it might be filtered
        if not desc:
            print("PROBLEM: No description!")
        elif len(desc) < 50:
            print(f"PROBLEM: Description too short ({len(desc)} < 50)")
        else:
            bad_phrases = [
                "unable to analyze", "can't analyze", "cannot analyze",
                "i'm unable", "i cannot", "provide a description",
            ]
            for phrase in bad_phrases:
                if phrase in desc.lower():
                    print(f"PROBLEM: Contains bad phrase: '{phrase}'")
                    break
            else:
                print("STATUS: Description looks OK!")
        print("-"*60)

if not found:
    print("\nNo VISAPP or ALGORITHM figures found in JSON!")