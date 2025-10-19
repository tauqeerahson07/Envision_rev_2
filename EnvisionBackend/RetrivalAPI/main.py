from typing import Dict, Any, List, Optional
import os
import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from.Services.checkpoints import checkpointer
from .Services.script_generator import(
    generate_script,
    detect_project_type,
    extractScenes,
    build_base_image_prompts,
)
from .Services.image_generator import FluxImageGenerator
from .Services.video_generator import VideoGenerator

load_dotenv()

State = Dict[str, Any]

# ---------- Nodes ----------
def node_generate_script(state: State) -> State:
    """Generate the initial script based on user concept."""
    concept = state["concept"]
    num_scenes = state["num_scenes"]
    creativity = state["creativity"]

    print("🎬 Generating script...")
    try:
        res = generate_script(concept, num_scenes, creativity)
        
        state["script"] = res["script"]
        state["characters"] = res["character_details"]
        state["scenes"] = res["scene_details"]
        state["product"] = res.get("product_details", {})
        state['temperature'] = res['temperature']
        state["project_type"] = res.get("project_type", detect_project_type(concept))

        print("\n✅ Script generated successfully!")
        print(f"\n📋 Characters ({len(state['characters'])}):")
        for name, details in state["characters"].items():
            print(f"  • {name}: {details.get('description', 'No description')}")
        
        print(f"\n🎭 Scenes ({len(state['scenes'])}):")
        for s in state["scenes"]:
            print(f"  • Scene {s['scene_number']}: {s['title']}")
            print(f"    Story: {s['story'][:100]}...")
            print(f"    Dialogue lines: {len(s.get('dialogue_lines', []))}")
        
        if state["project_type"] == "commercial":
            print("\n🛍️ Product Details:")
            for k, v in state["product"].items():
                print(f"  • {k}: {v}")
                
    except Exception as e:
        print(f"❌ Error generating script: {e}")
        state["error"] = str(e)
        
    return state

def node_decide_rewrite(state: State) -> State:
    scenes = state.get("scenes", [])
    if not scenes:
        state["scene_to_edit"] = None
        state["needs_rewrite"] = False
        return state

    # If decisions are not programmatic, pause here
    if "rewrite_decision" not in state:
        state["pause_for_decision"] = True
        return state

    # Existing logic for pre-set decisions
    decision = state["rewrite_decision"]
    if decision == "accept":
        state["needs_rewrite"] = False
        state["scene_to_edit"] = None
    elif decision == "edit":
        state["needs_rewrite"] = True
    return state

def node_decide_image_edit(state: State) -> State:
    scenes = state.get("scenes", [])
    if not scenes:
        state["scene_to_edit"] = None
        state["needs_image_edit"] = False
        return state

    # If decisions are not programmatic, pause here
    if "image_edit_decision" not in state:
        state["pause_for_decision"] = True
        return state

    # Existing logic for pre-set decisions
    decision = state["image_edit_decision"]
    if decision == "accept":
        state["needs_image_edit"] = False
        state["scene_to_edit"] = None
    elif decision == "edit":
        state["needs_image_edit"] = True
    return state

def node_rewrite_scene(state: State) -> State:
    """
    Rewrite a specific scene based on user instructions, maintaining character consistency.
    """
    import os
    import requests
    import re

    target = state.get("scene_to_edit")
    if not target:
        state["scene_to_edit"] = None
        return state

    scenes = state.get("scenes", [])
    scene_map = {s["scene_number"]: s for s in scenes}
    current = scene_map.get(target)
    if not current:
        state["scene_to_edit"] = None
        state["error"] = f"Scene {target} not found."
        return state

    user_notes = state.get("rewrite_instructions").strip()
    if not user_notes:
        state["scene_to_edit"] = None
        state["error"] = "No rewrite instructions provided."
        return state

    api_key = os.getenv("NEBIUS_API_KEY")
    api_base = os.getenv("NEBIUS_API_BASE")
    if not api_key or not api_base:
        state["scene_to_edit"] = None
        state["error"] = "Nebius API not configured."
        return state

    # Build character reference sheet for prompt
    character_sheet = ""
    for idx, (name, details) in enumerate(state.get("characters", {}).items(), 1):
        character_sheet += f"CHARACTER {idx}:\n"
        character_sheet += f"- Name: {name}\n"
        for key, value in details.items():
            character_sheet += f"- {key.capitalize()}: {value}\n"
        character_sheet += "\n"

    # System prompt for rewrite
    system_prompt = (
        "You are a master screenplay writer tasked with rewriting exactly one scene from the screenplay for animation rendering.\n"
        "🚨 CRITICAL CHARACTER CONSISTENCY RULES 🚨\n"
        "1. Start with a CHARACTER REFERENCE SHEET.\n"
        "2. Use IDENTICAL character descriptions (word-for-word) in every scene.\n"
        "3. DO NOT shorten, rephrase, or adapt the character description in any way.\n"
        "4. COPY-PASTE the exact same character description into each [Actors:] block.\n"
        "⚠️ If you skip the CHARACTER REFERENCE SHEET or use inconsistent names, the animation pipeline will break.\n"
        "Return ONLY the rewritten scene in this exact format:\n"
        "**Scene #: \"TITLE\"**\n"
        "[Actors: Full Name1, Full Name2, Full Name3]\n"
        "**Story of the scene**\n"
        "**Script:**\n"
        "Full Name1: (action) dialogue\n"
        "Full Name2: (action) dialogue\n"
        "..."
    )

    # User prompt for rewrite
    user_prompt = (
    f"CHARACTER REFERENCE SHEET:\n{character_sheet}\n"
    f"FULL SCREENPLAY FOR CONTEXT:\n{state.get('script', '')}\n\n"
    f"TARGET SCENE TO REWRITE: Scene {target}\n\n"
    f"CURRENT SCENE CONTENT:\n{current.get('script', '')}\n\n"
    f"USER REWRITE INSTRUCTIONS:\n{user_notes}\n\n"
    "WARNING: If you do not change the scene according to the EDIT REQUEST, your output will be rejected.\n"
    f"You MUST take the {user_notes} to edit the scene and make cohesion with rest of the scenes.\n"
    "You MUST incorporate the user's rewrite instructions clearly and visibly in the new scene."
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": state.get('temperature'),
        "max_tokens": 2000,
    }

    try:
        resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        print("\n📝 Proposed Rewrite:")
        print("-" * 50)
        print(content)
        print("-" * 50)

        rewritten_scenes = extractScenes(content)
        if not rewritten_scenes:
            state["error"] = "Could not parse rewritten scene. Content was: " + content
            state["scene_to_edit"] = None
            return state

        new_scene = rewritten_scenes[0]
        new_scene["scene_number"] = target  # Ensure correct scene number

        # Update the scenes list
        updated_scenes = []
        for s in state["scenes"]:
            if s["scene_number"] == target:
                updated_scenes.append(new_scene)
            else:
                updated_scenes.append(s)

        state["scenes"] = updated_scenes
        print(f"✅ Scene {target} successfully updated!")

        state["script_needs_update"] = True
        state["scene_to_edit"] = None
        
        # --- REGENERATE SUBSEQUENT SCENES FOR CONTEXT CONSISTENCY ---
        # Accumulate context up to the edited scene
        context_so_far = ""
        for s in updated_scenes:
            if s["scene_number"] <= target:
                context_so_far += f"\n**Scene {s['scene_number']}: \"{s['title']}\"**\n{s['story_context']}"
        # Regenerate all scenes after the edited one
        for idx, s in enumerate(updated_scenes):
            if s["scene_number"] > target:
                # Call your script generation function with context_so_far
                regen_result = generate_script(
                    state["concept"],
                    1,
                    state.get("creativity", "balanced"),
                    previous_context=context_so_far
                )
                if regen_result and regen_result.get("scene_details"):
                    regen_scene = regen_result["scene_details"][0]
                    regen_scene["scene_number"] = s["scene_number"]
                    updated_scenes[idx] = regen_scene
                    context_so_far += f"\n**Scene {regen_scene['scene_number']}: \"{regen_scene['title']}\"**\n{regen_scene['story_context']}"

        state["scenes"] = updated_scenes
        state["script_needs_update"] = True
        state["scene_to_edit"] = None

        return state

    except Exception as e:
        state["error"] = f"An unexpected error occurred during rewrite: {e}"
        state["scene_to_edit"] = None
        return state

def node_extract_prompts(state: State) -> State:
    """Extract image generation prompts from characters and scenes."""
    print("\n🎨 Extracting image prompts...")
    try:
        prompts = build_base_image_prompts(state.get("characters", {}))
        state["group_prompt"] = prompts.get("group_prompt", "")
        state["character_prompts"] = prompts.get("character_prompts", {})
        
        print("✅ Prompts extracted successfully!")
        print(f"📝 Group prompt preview: {state['group_prompt'][:100]}...")
        print(f"👥 Character prompts: {len(state['character_prompts'])} characters")
        
    except Exception as e:
        print(f"❌ Error extracting prompts: {e}")
        state["error"] = str(e)
        
    return state

def node_generate_images(state: State) -> State:
    """Generate images for all scenes."""
    print("\n🖼️ Generating images...")
    try:
        flux = FluxImageGenerator()
        chars = state.get("characters", {})
        scenes = state.get("scenes", [])
        product = state.get("product", {})
        project_type = state.get("project_type", "story")

        # Create base reference image first
        print("🎭 Creating base reference image...")
        if project_type == "commercial":
            base = flux.create_base_commercial_reference_image(chars, product)
        else:
            base = flux.create_base_character_image(chars)

        if not base:
            print("❌ Failed to create base reference image.")
            state["error"] = "Base image generation failed"
            return state

        state["base_image"] = base
        print("✅ Base reference image created!")

        # Generate scene images
        print(f"🎬 Generating images for {len(scenes)} scenes...")
        if project_type == "commercial":
            result = flux.generate_commercial_sequence(scenes, chars, product)
            state["image_result"] = result
            state["scene_images"] = result.get("scene_images")
            state["image_prompts"] = result.get("scene_prompt")
            state["product_images"] = result.get("product_images")
            state["commercial_frame_prompts"]= result.get("commercial_frame_prompts")
        else:
            state['image_prompts'],state["scene_images"] = flux.generate_scene_images(scenes=scenes, characters=chars)
            print(state['image_prompts'])

        print(f"✅ Generated {len(state.get('scene_images', []))} scene images!")
        
        state["needs_image_edit"] = True
    except Exception as e:
        print(f"❌ Error generating images: {e}")
        state["error"] = str(e)
        
    return state

def node_edit_images(state: State) -> State:
    """Edit images for all scenes or a specific scene based on user instructions."""
    print("\n🖼️ Editing images for the scenes...")

    scene_to_edit = state.get("scene_to_edit")  # Specific scene to edit (if provided)
    scenes = state.get("scenes", [])
    scene_map = {s["scene_number"]: s for s in scenes}  # Map scenes by scene_number
    edit_instructions = state.get("rewrite_instructions", "").strip()
    style = state.get("style", "").strip().lower()

    if not edit_instructions:
        state["error"] = "No edit instructions provided."
        return state

    print(f"🖌️ Applying style: {style}")
    if not edit_instructions:
        aggregated_prompt = (
            f"Edit the image to be in the following {style}. Incorporate these instructions: {edit_instructions} "
            f"!!!!! Important Ensure {style} of the image is applied on Image as requested by user !!!!"
        )
        edit_instructions = aggregated_prompt

    # Initialize the image generator
    image_generator = FluxImageGenerator()
    image_generator.base_style = style
    print(f"image_generator.base_style: {image_generator.base_style}")

    # If a specific scene is provided, edit only that scene
    if scene_to_edit:
        current_scene = scene_map.get(scene_to_edit)
        if not current_scene:
            state["error"] = f"Scene {scene_to_edit} not found in state."
            return state

        image = current_scene.get("scene_image")
        if not image:
            state["error"] = f"Scene {scene_to_edit} is missing image data."
            return state

        print(f"Editing scene {scene_to_edit} with style {style}")
        edited_image = image_generator._generate(edit_instructions, image)
        if not edited_image:
            state["error"] = f"Image editing failed for scene {scene_to_edit}."
            return state

        # Update the scene with the edited image and prompt
        current_scene["scene_image"] = edited_image
        current_scene["scene_image_prompt"] = edit_instructions
        print(f"✅ Scene {scene_to_edit} image edited successfully!")

    # If no specific scene is provided, edit all scenes
    else:
        print("Editing all scenes...")
        images = [s.get("scene_image") for s in scenes if s.get("scene_image")]
        for scene in scenes:
            scene_number = scene.get("scene_number")
            image = scene.get("scene_image")
            context = scene.get("story_context", "")
            if not image:
                print(f"❌ Scene {scene_number} is missing image data. Skipping...")
                continue
            
            if not context:
                print(f"❌ Scene {scene_number} is missing story context. Skipping...")
                continue

            print(f"Editing scene {scene_number} with style {style}")
            edited_image = image_generator._generateMultiple(context, images)
            if not edited_image:
                state["error"] = f"Image editing failed for scene {scene_number}."
                return state
            
            images.remove(image)  # Remove used image to avoid reusing
            images.append(edited_image)  # Add edited image for potential use in next iterations
            # Update the scene with the edited image and prompt
            scene["scene_image"] = edited_image
            scene["scene_image_prompt"] = aggregated_prompt
            print(f"✅ Scene {scene_number} image edited successfully!")
            
    state["needs_image_edit"] = True
    # Update the state with the modified scenes
    state["scenes"] = scenes
    state["scene_to_edit"] = None  # Clear the specific scene to edit
    print(f"✅ All scene images edited successfully!")
    return state

def node_generate_video(state: State) -> State:
    """Generate a video from the scene images."""
    print("\n🎥 Generating video from scene images...")
    try:
        video_gen = VideoGenerator()
        scenes = state.get("scenes", [])

        if not scenes:
            state["error"] = "No scenes available for video generation."
            return state

        videos = []

        for scene in scenes:
            scene_number = scene["scene_number"]
            image = scene["image"]
            prompt = scene["prompt"]

            if not image:
                print(f"❌ Scene {scene_number} is missing image data. Skipping...")
                continue

            if not prompt:
                print(f"❌ Scene {scene_number} is missing prompt data. Skipping...")
                continue

            print(f"🎬 Generating video for scene {scene_number}...")
            video = video_gen.generate_video(prompt, image)

            if not video:
                print(f"❌ Video generation failed for scene {scene_number}.")
                continue

            videos.append({"scene_number": scene_number, "video": video})
            print(f"✅ Video generated for scene {scene_number} successfully!")

        state["scene_videos"] = videos
        print(f"✅ Generated {len(videos)} videos successfully!")

    except Exception as e:
        print(f"❌ Error generating video: {e}")
        state["error"] = str(e)

    return state


def node_finalize_output(state: State) -> State:
    """Finalize and present the complete output."""
    print("\n" + "="*60)
    print("🎉 PROJECT COMPLETED!")
    print("="*60)
    
    print(f"\n📊 PROJECT SUMMARY:")
    print(f"  • Concept: {state['concept']}")
    print(f"  • Project Type: {state.get('project_type', 'Unknown')}")
    print(f"  • Total Scenes: {len(state.get('scenes', []))}")
    print(f"  • Characters: {len(state.get('characters', {}))}")
    print(f"  • Images Generated: {len(state.get('scene_images', []))}")
    
    if state.get("error"):
        print(f"\n⚠️ Errors encountered: {state['error']}")
    
    # Save outputs if needed
    try:
        # You could add file saving logic here
        print(f"\n💾 All content ready for export!")
    except Exception as e:
        print(f"⚠️ Note: {e}")
    
    return state

# ---------- Routing Functions ----------
def route_after_decide(state: State) -> str:
    """Route based on whether user wants to rewrite scenes."""
    if state.get("needs_rewrite", False):
        return "rewrite_scene"
    else:
        return "extract_prompts"

def route_after_rewrite(state: State) -> str:
    """Route after scene rewrite - either continue editing or proceed."""
    if state.get("needs_rewrite", False):
        return "decide_rewrite"  # Loop back to allow editing more scenes
    else:
        return "extract_prompts"
    
def route_after_generate_images(state: State) -> str:
    """Route based on whether image edits are required."""
    if state.get("needs_image_edit", False):
        return "decide_image_edit"  
    else:
        return "generate_video"

# ---------- Workflow Builder ----------
def build_workflow(entry_point="generate_script"):
    """Build and return the complete LangGraph workflow."""
    g = StateGraph(State)
    
    # Add all nodes
    g.add_node("generate_script", node_generate_script)
    g.add_node("decide_rewrite", node_decide_rewrite)
    g.add_node("rewrite_scene", node_rewrite_scene)
    g.add_node("extract_prompts", node_extract_prompts)
    g.add_node("generate_images", node_generate_images)
    g.add_node("decide_image_edit", node_decide_rewrite)  
    g.add_node("edit_images", node_edit_images)  
    g.add_node("generate_video", node_generate_video)
    g.add_node("finalize_output", node_finalize_output)

    # Set entry point
    g.set_entry_point(entry_point)
    
    # Add edges
    g.add_edge("generate_script", "decide_rewrite")
    g.add_conditional_edges(
        "decide_rewrite",
        route_after_decide,
        {
            "rewrite_scene": "rewrite_scene",
            "extract_prompts": "extract_prompts"
        },
    )
    g.add_conditional_edges(
        "rewrite_scene",
        route_after_rewrite,
        {
            "decide_rewrite": "decide_rewrite",
            "extract_prompts": "extract_prompts"
        },
    )
    g.add_edge("extract_prompts", "generate_images")
    g.add_conditional_edges(
        "generate_images",
        route_after_generate_images,
        {
            "decide_image_edit": "decide_image_edit",
            "generate_video": "generate_video"
        },
    )

    g.add_edge("generate_video", END)
    # g.add_edge("finalize_output", END)
    
    return g.compile(checkpointer=checkpointer)

def validate_inputs(concept: str, num_scenes: str, creativity: str) -> tuple[str, int, str]:
    """Validate and normalize user inputs."""
    # Validate concept
    if not concept.strip():
        raise ValueError("Concept cannot be empty")
    
    # Validate num_scenes
    try:
        num_scenes_int = int(num_scenes) if num_scenes.strip() else 5
        if num_scenes_int < 1 or num_scenes_int > 7:
            print("⚠️ Number of scenes should be between 1-7. Using default: 5")
            num_scenes_int = 5
    except ValueError:
        print("⚠️ Invalid number of scenes. Using default: 5")
        num_scenes_int = 5
    
    # Validate creativity
    valid_creativity = ["factual", "creative", "balanced"]
    creativity_clean = creativity.strip().lower() if creativity.strip() else "balanced"
    if creativity_clean not in valid_creativity:
        print(f"⚠️ Invalid creativity level. Using default: balanced")
        creativity_clean = "balanced"
    
    return concept.strip(), num_scenes_int, creativity_clean
