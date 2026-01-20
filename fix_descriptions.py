"""
Re-describe figures that have invalid descriptions.
"""
import json
import base64
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DESCRIPTIONS_FILE = Path("data/figure_descriptions.json")

def is_valid_description(desc: str) -> bool:
    if not desc or len(desc) < 50:
        return False
    
    bad_phrases = [
        "unable to analyze", "can't analyze", "cannot analyze",
        "i'm unable", "i cannot", "provide a description",
        "provide more context", "describe it or provide",
        "i can't directly", "cannot directly", "if you describe it",
        "i'm not able", "cannot view", "can't view",
    ]
    
    desc_lower = desc.lower()
    return not any(phrase in desc_lower for phrase in bad_phrases)


def describe_figure(image_path: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    ext = Path(image_path).suffix.lower()
    media_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Using gpt-4o instead of gpt-4o-mini for better results
        messages=[
            {
                "role": "system",
                "content": "You are a computer vision expert. You MUST describe every image provided. Describe the architecture, components, and data flow visible in the diagram. Never refuse to describe an image."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Describe this architecture diagram from a computer vision paper.

IMPORTANT: You must provide a description. Focus on:
- Main components visible (encoder, decoder, networks, modules)
- Architecture type (CNN, GCN, Transformer, etc.)
- Data flow between components
- Any text labels visible in the diagram

Provide a 2-3 sentence technical description."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                            "detail": "high"  # Using high detail for better analysis
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content


def main():
    # Load figures
    with open(DESCRIPTIONS_FILE, 'r') as f:
        figures = json.load(f)
    
    print("="*60)
    print("RE-DESCRIBING INVALID FIGURES")
    print("="*60)
    
    updated = 0
    for fig in figures:
        desc = fig.get('description', '')
        
        if not is_valid_description(desc):
            filepath = fig.get('filepath', '')
            filename = fig.get('filename', '')
            
            print(f"\nRe-describing: {filename}")
            print(f"  Old description: {desc[:100]}...")
            
            if Path(filepath).exists():
                try:
                    new_desc = describe_figure(filepath)
                    fig['description'] = new_desc
                    print(f"  New description: {new_desc[:100]}...")
                    
                    if is_valid_description(new_desc):
                        print("  STATUS: Success!")
                        updated += 1
                    else:
                        print("  STATUS: Still invalid")
                except Exception as e:
                    print(f"  ERROR: {e}")
            else:
                print(f"  ERROR: File not found: {filepath}")
    
    # Save updated figures
    if updated > 0:
        with open(DESCRIPTIONS_FILE, 'w') as f:
            json.dump(figures, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Updated {updated} figures")
        print(f"Saved to {DESCRIPTIONS_FILE}")
        print(f"\nNow run: python src/index_figures.py")
    else:
        print("\nNo figures were updated.")


if __name__ == "__main__":
    main()