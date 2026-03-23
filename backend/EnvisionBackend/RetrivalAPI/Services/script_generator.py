import os
import requests
import re
from typing import Dict, List, Any, TypedDict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Nebius API credentials
NEBIUS_API_KEY = os.getenv('NEBIUS_API_KEY')
NEBIUS_API_BASE = os.getenv('NEBIUS_API_BASE')

if not NEBIUS_API_KEY:
    raise ValueError("Nebius_key not found in environment variables")

# Model to use
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Define Scene structure
class SceneData(TypedDict):
    scene_number: int
    title: str
    actors: List[str]
    story: str
    script: str
    dialogue_lines: List[Dict[str, str]]  

# Enhanced WorkflowState for type hints
class WorkflowState(TypedDict):
    concept: str
    num_scenes: int
    creativity_level: str
    script: str
    characters: Dict[str, Dict[str, str]]
    scenes: List[SceneData]
    group_prompt: str
    character_prompts: Dict[str, str]
    scene_to_edit: Optional[int]
    scene_prompt: Optional[str]
    image_prompts: List[str]
    scene_images: List[List[str]]
    errors: List[str]
    status: str
    project_type: str

# ...existing code...
def extractCharacters(script: str) -> Dict[str, Dict[str, str]]:
    """
    Extract character details from the script text and return in the required format:
    Dict[str, Dict[str, str]]
    Supports both:
      - 'CHARACTER 1:' block style
      - Bullet list with '- **Name:** ...' markdown formatting
    """
    characters: Dict[str, Dict[str, str]] = {}

    # Locate character section
    ref_sheet_start = script.find("**CHARACTER REFERENCE SHEET**")
    if ref_sheet_start == -1:
        ref_sheet_start = script.find("CHARACTER REFERENCE SHEET")
    if ref_sheet_start == -1:
        ref_sheet_start = script.find("CHARACTER 1:")
        if ref_sheet_start == -1:
            ref_sheet_start = script.find("STEP 1:")

    # Find start of scenes to delimit character section
    scene_start = script.find("**Scene")
    if scene_start == -1:
        scene_start = script.find("Scene 1:")
    if scene_start == -1:
        scene_start = script.find("STEP 2:")
    if scene_start == -1:
        scene_start = len(script)

    if ref_sheet_start == -1:
        return characters  # Not found

    char_section = script[ref_sheet_start:scene_start]

    # First try CHARACTER X: style
    char_blocks = re.split(r'CHARACTER\s+\d+:', char_section, flags=re.IGNORECASE)

    if len(char_blocks) > 1:
        for block in char_blocks:
            if not block.strip():
                continue
            char_data = {}
            attributes = {
                'name': r'-?\s*\**Name\**:\s*([^\n]+)',
                'gender': r'-?\s*\**Gender\**:\s*([^\n]+)',
                'hair': r'-?\s*\**Hair\**:\s*([^\n]+)',
                'face': r'-?\s*\**Face\**:\s*([^\n]+)',
                'eyes': r'-?\s*\**Eyes\**:\s*([^\n]+)',
                'skin': r'-?\s*\**Skin\**:\s*([^\n]+)',
                'build': r'-?\s*\**Build\**:\s*([^\n]+)',
                'outfit': r'-?\s*\**Outfit\**:\s*([^\n]+)',
                'distinctive': r'-?\s*\**Distinctive\**:\s*([^\n]+)'
            }
            for attr, pattern in attributes.items():
                m = re.search(pattern, block, re.IGNORECASE)
                if m:
                    char_data[attr] = m.group(1).strip().rstrip('.')
            if 'name' in char_data:
                name_clean = char_data['name'].strip('*').strip()
                char_data['name'] = name_clean
                characters[name_clean] = char_data
        if characters:
            return characters  # Successfully parsed this style

    # Fallback: bullet list style with "- **Field:** value"
    lines = [l.rstrip() for l in char_section.splitlines()]
    current_char: Dict[str, str] = {}
    current_name: Optional[str] = None

    def commit():
        if current_name and current_char:
            name_clean = current_name.strip('*').strip()
            current_char['name'] = name_clean
            characters[name_clean] = current_char.copy()

    bullet_attr_pattern = re.compile(
        r'^-?\s*\**(Name|Gender|Hair|Face|Eyes|Skin|Build|Outfit|Distinctive)\**:\s*(.+)$',
        re.IGNORECASE
    )

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('**') or line.startswith('---'):
            continue

        # Remove leading dash if present
        if line.startswith('- '):
            line = line[2:].strip()

        # Try match attribute
        m = bullet_attr_pattern.match(line.replace('**', ''))
        if m:
            attr = m.group(1).lower()
            value = m.group(2).strip()
            # New character start
            if attr == 'name':
                if current_name:
                    commit()
                current_char = {'name': value}
                current_name = value
            else:
                if current_char is not None:
                    current_char[attr] = value
        else:
            # Non-matching line inside character block (rare) – ignore
            continue

    # Commit last
    if current_name:
        commit()

    return characters

def extract_product_details(script: str) -> Dict[str, str]:
    """
    Extract product metadata from the script if STEP 3: PRODUCT DETAILS is present.
    """
    product = {}

    # Look for STEP 3: PRODUCT DETAILS section
    step3_start = script.find("**STEP 3: PRODUCT DETAILS**")
    if step3_start == -1:
        step3_start = script.find("STEP 3: PRODUCT DETAILS")
    if step3_start == -1:
        step3_start = script.find("**PRODUCT DETAILS**")
    
    if step3_start != -1:
        # Extract everything after the product details header
        section = script[step3_start:]
        
        # Split into lines for processing
        lines = section.split('\n')
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('**') or line.startswith('---'):
                continue
            
            # Look for field patterns
            if line.startswith('- **Name:**') or line.startswith('- Name:'):
                value = line.replace('- **Name:**', '').replace('- Name:', '').strip()
                # Clean up any markdown formatting
                value = value.replace('**', '').strip()
                if value:
                    product['name'] = value
                    
            elif line.startswith('- **Type:**') or line.startswith('- Type:'):
                value = line.replace('- **Type:**', '').replace('- Type:', '').strip()
                value = value.replace('**', '').strip()
                if value:
                    product['type'] = value
                    
            elif line.startswith('- **Features:**') or line.startswith('- Features:'):
                # For features, we need to collect multiple lines
                value = line.replace('- **Features:**', '').replace('- Features:', '').strip()
                value = value.replace('**', '').strip()
                
                # Collect subsequent feature lines
                features_list = []
                if value:  # If there's content on the same line
                    features_list.append(value)
                
                # Look ahead for more feature lines (indented with spaces or dashes)
                for next_line in lines[line_idx + 1:]:
                    next_line = next_line.strip()
                    if not next_line:
                        continue
                    if next_line.startswith('- **') or (next_line.startswith('- ') and ':' in next_line):
                        break  # Next field found
                    if next_line.startswith('-') or next_line.startswith('  '):
                        # This is a feature bullet point
                        feature = next_line.lstrip('- ').strip()
                        if feature:
                            features_list.append(feature)
                
                if features_list:
                    product['features'] = '; '.join(features_list)
                    
            elif line.startswith('- **Visual Style:**') or line.startswith('- Visual Style:'):
                value = line.replace('- **Visual Style:**', '').replace('- Visual Style:', '').strip()
                value = value.replace('**', '').strip()
                if value:
                    product['visual style'] = value
                    
            elif line.startswith('- **Brand Feel:**') or line.startswith('- Brand Feel:'):
                value = line.replace('- **Brand Feel:**', '').replace('- Brand Feel:', '').strip()
                value = value.replace('**', '').strip()
                if value:
                    product['brand feel'] = value

    return product


def extractScenes(script: str) -> List[SceneData]:
    scenes = []
    # This pattern matches scenes with flexible story/script separation and optional bold/colon
    scene_pattern = (
        r'\*\*Scene\s+(\d+):\s*"([^"]+)"\*\*\s*'                # Scene number & title
        r'\[Actors:\s*([^\]]+)\]\s*'                            # Actors list
        r'\*\*Story of the scene:?[\*\s]*([^\n]+)\n+'           # Story text (single line, after colon)
        r'\*\*Script:\*\*\s*([\s\S]*?)(?=\n\s*\*\*Scene|\Z)'    # Script text (until next scene or end)
    )
    matches = re.findall(scene_pattern, script, re.IGNORECASE)
    for match in matches:
        scene_number = int(match[0])
        title = match[1].strip()
        actors_text = match[2].strip()
        story = match[3].strip()
        script_text = match[4].strip()
        actors = [actor.strip() for actor in actors_text.split(',')]
        dialogue_lines = []
        dialogue_pattern = r'([A-Za-z ."\']+):\s*(?:\(([^)]*)\))?\s*([^\n]+)'
        for d_match in re.findall(dialogue_pattern, script_text):
            dialogue_lines.append({
                "character": d_match[0].strip(),
                "action": d_match[1].strip(),
                "dialogue": d_match[2].strip()
            })
        scenes.append({
            "scene_number": scene_number,
            "title": title,
            "actors": actors,
            "story_context": story,
            "script": script_text,
            "dialogue_lines": dialogue_lines
        })
    return scenes

def generate_script(concept: str, num_scenes: int = 5, creativity_level: str = 'balanced',previous_context: str = "") -> Dict[str, Any]:
    try:
        if creativity_level == "factual":
            temperature = 0.5
            description = "factual and realistic"
        elif creativity_level == "creative":
            temperature = 0.9
            description = "creative and imaginative"
        else:
            temperature = 0.7
            description = "balanced blend of realism and creativity"

        include_product_step = False 
        project_type = "story"
        if detect_project_type(concept) == 'commercial':
            project_type = 'commercial'
            include_product_step = True

        # System prompt
        system_prompt = """You are a master screenplay writer tasked with creating highly detailed, visually consistent scripts where every character remains 100% identical across all scenes for animation rendering.

🚨 CRITICAL CHARACTER CONSISTENCY RULES 🚨
1. Start with a CHARACTER REFERENCE SHEET.
2. Use IDENTICAL character descriptions (word-for-word) in every scene.
3. DO NOT shorten, rephrase, or adapt the character description in any way.
4. COPY-PASTE the exact same character description into each [Actors:] block.
⚠️ If you skip the CHARACTER REFERENCE SHEET or use inconsistent names, the animation pipeline will break.

✅ CHARACTER DESCRIPTION TEMPLATE (USE THIS EXACT FORMAT):
- Name: [Full Name]
- Gender: [e.g., Male, Female]
- Hair: [exact color, length, style, texture]
- Face: [shape, features, expression]
- Eyes: [color, shape, distinctive features]
- Skin: [tone, texture, marks or scars]
- Build: [height, body type, posture]
- Outfit: [specific clothing items, colors, accessories]
- Distinctive: [tattoos, scars, birthmarks, jewelry, etc.]

✅ SCENE FORMAT (USE EXACTLY):
**Scene #: "TITLE"**
[Actors: Full Name1, Full Name2, Full Name3]
**Story of the scene**
**Script:**
Full Name1: (action) dialogue  
Full Name2: (action) dialogue
"""
        
        # Generation prompt with conditional STEP 3
        generation_prompt = f"""
You must generate a screenplay titled "{concept}" in two sections:

---

STEP 1: CHARACTER REFERENCE SHEET  
Create exactly 3 characters using the exact format below. Do NOT skip this step.

CHARACTER 1:
- Name: [Full Name]
- Gender: ...
- Hair: ...
- Face: ...
- Eyes: ...
- Skin: ...
- Build: ...
- Outfit: ...
- Distinctive: ...

CHARACTER 2:
- Name: [Full Name]
- Gender: ...
- Hair: ...
- Face: ...
- Eyes: ...
- Skin: ...
- Build: ...
- Outfit: ...
- Distinctive: ...

CHARACTER 3:
- Name: [Full Name]
- Gender: ...
- Hair: ...
- Face: ...
- Eyes: ...
- Skin: ...
- Build: ...
- Outfit: ...
- Distinctive: ...

---

STEP 2: SCRIPT SCENES  
Create exactly {num_scenes} scenes. Each scene MUST include:
- [Actors: Full Name1, Full Name2, Full Name3] (using exact names from above)
- A short story description of the scene
- A full dialogue script, using character names and actions

Format:
**Scene 1: "Title of Scene"**  
[Actors: Full Name1, Full Name2, Full Name3]  
**Story of the scene**  
**Script:**  
Full Name1: (action) dialogue  
Full Name2: (action) dialogue  
...
"""

        if include_product_step:
            generation_prompt += """
---

STEP 3: PRODUCT DETAILS  
List the product being advertised and describe it in detail including:
- Name:  
- Type:  
- Features:  
- Visual Style:  
- Brand Feel:
"""
        if previous_context:
            generation_prompt = (
                f"CONTEXT FROM PREVIOUS SCENES:\n{previous_context}\n\n"
                + generation_prompt
            )

        headers = {
            "Authorization": f"Bearer {NEBIUS_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": generation_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 4000
        }

        response = requests.post(
            f"{NEBIUS_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )
        if response.status_code == 200:
            result = response.json()
            script_text = result["choices"][0]["message"]["content"]
            character_details = extractCharacters(script_text)
            scene_details = extractScenes(script_text)
            print(scene_details)
            product_details = extract_product_details(script_text) if include_product_step else {}

            return {
                "script": script_text, 
                "temperature":temperature,
                "character_details": character_details,
                "scene_details": scene_details,
                "product_details": product_details,
                "project_type": project_type
            }
        else:
            return {
                "script": f"API Error {response.status_code}: {response.text}", 
                "character_details": {},
                "scene_details": [],
                "product_details": {},
                "temperature":temperature,
                "project_type": project_type
            }

    except Exception as e:
        return {
            "temperature":temperature,
            "script": f"Error generating script: {str(e)}", 
            "character_details": {},
            "scene_details": [],
            "product_details": {},
            "project_type": project_type
        }

def detect_project_type(concept: str) -> str:
    """
    Infer if the project is a 'story' or 'commercial' based on concept and script content.
    """
    concept_lower = concept.lower()

    commercial_keywords = ["advert", "advertisement", "commercial", "promo", "promotional", "product", "sale", "buy now", "features"]

    for keyword in commercial_keywords:
        if keyword in concept_lower:
            return "commercial"

    return "story"

def build_base_image_prompts(character_details: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Build image prompts from character details with safe access"""
    if not character_details:
        return {"group_prompt": "", "character_prompts": {}}

    group_descriptions = []
    for char_name, char_data in character_details.items():
        # Use character name as fallback if 'name' key doesn't exist
        name = char_data.get('name', char_name)
        desc = (f"{name} (Gender: {char_data.get('gender', 'Unknown')}, "
                f"Hair: {char_data.get('hair', 'Unknown')}, "
                f"Face: {char_data.get('face', 'Unknown')}, "
                f"Eyes: {char_data.get('eyes', 'Unknown')}, "
                f"Skin: {char_data.get('skin', 'Unknown')}, "
                f"Build: {char_data.get('build', 'Unknown')}, "
                f"Outfit: {char_data.get('outfit', 'Unknown')}, "
                f"Distinctive: {char_data.get('distinctive', 'None')})")
        group_descriptions.append(desc)

    group_prompt = "A group image featuring: " + "; ".join(group_descriptions)

    character_prompts = {}
    for char_name, char_data in character_details.items():
        name = char_data.get('name', char_name)
        prompt = (f"{name} (Gender: {char_data.get('gender', 'Unknown')}, "
                  f"Hair: {char_data.get('hair', 'Unknown')}, "
                  f"Face: {char_data.get('face', 'Unknown')}, "
                  f"Eyes: {char_data.get('eyes', 'Unknown')}, "
                  f"Skin: {char_data.get('skin', 'Unknown')}, "
                  f"Build: {char_data.get('build', 'Unknown')}, "
                  f"Outfit: {char_data.get('outfit', 'Unknown')}, "
                  f"Distinctive: {char_data.get('distinctive', 'None')})")
        character_prompts[name] = prompt

    return {
        "group_prompt": group_prompt,
        "character_prompts": character_prompts
    }