import os
import requests
import json
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import base64
from PIL import Image
from io import BytesIO
import math
import random
import replicate
from replicate import Client

# Load environment variables
load_dotenv()

# FLUX API credentials
BFL_API = os.getenv('BLACKFOREST_API_KEY')
CREATION_URL = "https://api.bfl.ai/v1/flux-kontext-pro"
GET_IMAGE = "https://api.bfl.ai/v1/get_result"
REPLICATE_API = os.getenv('REPLICATE_API_TOKEN')

if not BFL_API:
    raise ValueError("BLACKFOREST_API_KEY not found in environment variables")

class FluxImageGenerator:
    """
    FLUX1.KONTEXT Image Generator for Character Consistent Scene Generation

    Enhancements added:
      - Professional photography styling baked into prompts (camera, lens, lighting)
      - Strong commercial prompts (product pedestal, reflections, bokeh)
      - Locked character descriptions for iterative consistency
      - Better frame-specific prompts for story stills
    """

    def __init__(
        self,
        style: Optional[str] = None,
        generation_delay: float = 1.5,
        retries: int = 2,
        default_aspect: str = "16:9",
        apply_filters: bool = False,
    ):
        """
        Args:
            style: optional extra style string to append to prompts
            generation_delay: seconds between API calls
            retries: number of times to retry the API call on transient failure
            default_aspect: default aspect ratio to request from the API
        """
        self.headers = {
            "x-key": BFL_API,
            "Content-Type": "application/json"
        }

        # A professional photography base style applied to all prompts
        # Camera/lens/lighting/postprocessing cues make AI produce more realistic, consistent results
        self.base_style = (
            "award-winning commercial photography, ultra high resolution, photorealistic, "
            "shot on Canon EOS R5 with 85mm lens (portrait) or 35mm (environment), "
            "use appropriate lens for the composition, realistic depth of field, f/1.4-f/4 depending on shot, "
            "three-point lighting for studio shots, golden hour or cinematic soft light for outdoors, "
            "accurate skin tones, correct anatomy, natural shadows and reflections, no distortions, no artifacts, "
            "clean background when required, polished color grading, HDR balance, cinematic composition"
        )

        if style:
            # allow user-supplied style to augment base_style
            self.base_style = f"{self.base_style}, {style}"

        self.generation_delay = generation_delay
        self.retries = max(0, int(retries))
        self.default_aspect = default_aspect

        # Will hold base images and character locks
        self.base_character_image: Optional[str] = None  # base64-encoded
        self.locked_character_descriptions: List[str] = []
        self.last_generated_urls: List[str] = []  # for debug or inspection

    # ---------- Low-level generate helper ----------
    def _generate_single_image(self, prompt: str, reference_image: str = None, seed: Optional[int] = None, aspect_ratio: Optional[str] = None) -> Optional[str]:
        """Generate a single image with optional reference image using the FLUX API.
           Returns a direct URL to the generated image sample when ready, or None on failure.
        """
        if aspect_ratio is None:
            aspect_ratio = self.default_aspect

        if seed is None:
            # produce randomized seed for variation when not provided
            seed = random.randint(1000, 999999)

        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "jpeg",
            "prompt_upsampling": False,
            "safety_tolerance": 2,
            "seed": int(seed)
        }

        if reference_image:
            payload = {
            "prompt": prompt,
            "input_image" : reference_image,
            "aspect_ratio": aspect_ratio,
            "output_format": "jpeg",
            "prompt_upsampling": False,
            "safety_tolerance": 2,
            "seed": int(seed)
        }

        attempt = 0
        while attempt <= self.retries:
            try:
                resp = requests.post(CREATION_URL, json=payload, headers=self.headers)
                if resp.status_code != 200:
                    attempt += 1
                    time.sleep(1 + attempt)
                    # If final attempt, print debug
                    if attempt > self.retries:
                        print(f"FLUX API error: {resp.status_code} - {resp.text}")
                        return None
                    continue

                data = resp.json()
                request_id = data.get("id")
                polling_url = data.get("polling_url")  # Add this line
                if not request_id:
                    print("FLUX response missing request id:", data)
                    return None

                # Poll for completion
                start_ts = time.time()
                poll_attempts = 0
                while True:
                    time.sleep(1.0)
                    poll_attempts += 1
                    # result_resp = requests.get(GET_IMAGE, headers=self.headers, params={"id": request_id})
                    result_resp = requests.get(polling_url, headers=self.headers)
                    if result_resp.status_code != 200:
                        # transient poll error - continue polling a few times
                        if poll_attempts > 10:
                            print("Polling error:", result_resp.status_code, result_resp.text)
                            break
                        else:
                            continue

                    result = result_resp.json()
                    status = result.get('status', 'Unknown')
                    if status == "Ready":
                        sample = result.get("result", {}).get("sample")
                        if sample:
                            # keep track of generated urls
                            self.last_generated_urls.append(sample)
                            return sample
                        else:
                            print("No sample in result:", result)
                            return None
                    elif status in ["Error", "Failed"]:
                        print("Generation failed:", result)
                        return None
                    # safety to avoid infinite loops
                    if time.time() - start_ts > 120:
                        print("Polling timed out for request:", request_id)
                        return None

            except requests.RequestException as re:
                attempt += 1
                print(f"Request exception (attempt {attempt}/{self.retries}): {re}")
                time.sleep(1 + attempt)
            except Exception as e:
                print("Unexpected exception during generation:", e)
                return None

        return None
    def _generate(self, prompt: str, reference_image: str = None, seed: Optional[int] = None, aspect_ratio: Optional[str] = None) -> Optional[str]:
        """
        Generate an image using Google Gemini with an optional reference image.
        Returns the image URL and appends it to self.last_generated_urls.
        """
        client = Client(api_token=REPLICATE_API)
        
        model = "google/nano-banana"  # or any other Replicate model you want

        input_data = {
            "prompt": prompt,
            "output_format": "jpg"
        }

        # Optionally include a reference image
        if reference_image:
            input_data["image_input"] = [reference_image]

        print("🚀 Generating image...")
        output = client.run(model, input=input_data)

        # The output is usually a list of URLs
        if output :
            if isinstance(output, list):
                image_url = str(output[0])
            else:
                image_url = str(output)
        else:
            print("❌ Failed to generate image")
            return None
        image =self._download_and_encode_image(image_url)
        return image
    def _generateMultiple(self, prompt: str, reference_images : str, seed: Optional[int] = None, aspect_ratio: Optional[str] = None) -> Optional[str]:
        """
        Generate an image using Google Gemini with an optional reference image.
        Returns the image URL and appends it to self.last_generated_urls.
        """
        client = Client(api_token=REPLICATE_API)
        
        model = "google/nano-banana"  # or any other Replicate model you want

        input_data = {
            "prompt": prompt,
            "output_format": "jpg"
        }

        # Optionally include a reference image
        if reference_images:
            for image in reference_images:
                input_data["image_input"] = [image]

        print("🚀 Generating image...")
        output = client.run(model, input=input_data)

        # The output is usually a list of URLs
        if output :
            if isinstance(output, list):
                image_url = str(output[0])
            else:
                image_url = str(output)
        else:
            print("❌ Failed to generate image")
            return None
        image =self._download_and_encode_image(image_url)
        return image

    # ---------- Utility: download and encode ----------
    def _download_and_encode_image(self, image_url: str) -> Optional[str]:
        """Download an image from a URL or handle a Base64-encoded string."""
        try:
            # Check if the input is already a Base64 string
            if image_url.startswith("data:image/"):
                print("✅ Input is already a Base64-encoded image.")
                return image_url  # Return the Base64 string as-is
    
            # Otherwise, treat it as a URL and download the image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            encoded = base64.b64encode(response.content).decode('utf-8')
            result = f"data:image/jpeg;base64,{encoded}"
            print("✅ Image downloaded and encoded:", image_url)
            return result
        except Exception as e:
            print(f"❌ Error downloading/encoding image: {e}")
            return None

    # ---------- Base character group portrait ----------
    def create_base_character_image(self, characters: Dict[str, Dict[str, str]], camera_hint: str = "85mm", background: str = "neutral seamless paper") -> Optional[str]:
        """
        Create a single base group portrait with all characters (character reference image).
        Saves locked character descriptions to self.locked_character_descriptions (list of strings).
        Returns base64 image string or None.
        """
        print(f"🎯 Creating base group portrait with {len(characters)} characters...")
        if not characters:
            print("❌ No characters provided.")
            return None

        character_descriptions = []
        for char_key, char_data in characters.items():
            name = char_data.get('name', char_key).strip('*').strip()
            # Build canonical locked description - consistent formatting
            gender = char_data.get('gender', 'unspecified')
            hair = char_data.get('hair', 'unspecified hair')
            eyes = char_data.get('eyes', 'unspecified eyes')
            face = char_data.get('face', 'unspecified face')
            outfit = char_data.get('outfit', 'unspecified outfit')
            distinctive = char_data.get('distinctive', '')
            parts = [f"{gender}", f"{hair}", f"{eyes} eyes", f"{face}", f"wearing {outfit}"]
            if distinctive:
                parts.append(distinctive)
            locked = f"{name} – " + ", ".join([p for p in parts if p])
            character_descriptions.append(locked)

        # persist locked descriptions for later use
        self.locked_character_descriptions = character_descriptions.copy()

        # Compose prompt with strong photographic cues
        group_prompt = (
            f"Group portrait: {', '.join(character_descriptions)}. "
            f"All subjects front-facing, evenly spaced, full-body visible, clear separation between each person. "
            f"Background: {background}. "
            f"Lighting: soft three-point studio lighting for flattering, even faces. "
            f"Camera: capture with {camera_hint} equivalent (Canon EOS R5 recommended), wide but showing full bodies. "
            f"{self.base_style}. "
            f"Natural, neutral expressions, high detail on faces, consistent clothing and features."
        )

        print(f"📸 Base image prompt (truncated): {group_prompt[:160]}...")
        # image_url = self._generate_single_image(group_prompt, seed=1000
        image_url = self._generate_single_image(group_prompt)
        if image_url:
            print("✅ Base group portrait generated successfully")
            base64_image = self._download_and_encode_image(image_url)
            if base64_image:
                self.base_character_image = base64_image
                return base64_image
            else:
                print("❌ Failed to download/encode base image")
                return None
        else:
            print("❌ Failed to generate base group portrait")
            return None

    # ---------- Commercial base (characters + product) ----------
    def create_base_commercial_reference_image(self, characters: Dict[str, Dict[str, str]], product_info: Dict[str, str], background: str = "neutral gradient") -> Optional[str]:
        """
        Create a single high-end commercial reference image: characters + hero product.
        Stores locked character descriptions and returns base64-encoded combined image.
        """
        print("🎯 Creating commercial reference image with characters and product...")
        if not characters or not product_info:
            print("❌ Missing characters or product info.")
            return None

        # product fields with defaults
        product_name = product_info.get('name', 'the product')
        product_type = product_info.get('type', 'product')
        product_material = product_info.get('material', 'premium materials')
        product_color = product_info.get('color', 'brand color')
        product_features = product_info.get('features', '')
        visual_style = product_info.get('visual style', 'modern, sleek')
        brand_feel = product_info.get('brand feel', 'premium, professional')

        # Build locked character descriptions in canonical form
        character_descriptions = []
        for char_key, char_data in characters.items():
            name = char_data.get('name', char_key).strip('*').strip()
            gender = char_data.get('gender', 'unspecified')
            hair = char_data.get('hair', 'unspecified hair')
            eyes = char_data.get('eyes', 'unspecified eyes')
            face = char_data.get('face', 'unspecified face')
            outfit = char_data.get('outfit', 'unspecified outfit')
            distinctive = char_data.get('distinctive', '')
            desc_parts = [f"{gender}", f"{hair}", f"{eyes} eyes", f"{face}", f"wearing {outfit}"]
            if distinctive:
                desc_parts.append(distinctive)
            locked = f"{name} – " + ", ".join([p for p in desc_parts if p])
            character_descriptions.append(locked)

        # persist locked descriptions
        self.locked_character_descriptions = character_descriptions.copy()

        # Compose commercial prompt with strong product centric cues
        prompt = (
            f"Luxury advertising group portrait featuring {', '.join(character_descriptions)}. "
            f"Composition: symmetric, characters positioned in the background, evenly spaced, slightly out of focus. "
            f"Hero: {product_name}, a {product_type} made of {product_material} in {product_color}, placed center foreground on a glossy black acrylic pedestal. "
            f"Product detail: {product_features}. "
            f"Lighting: dedicated hero key light above product, softbox fill left and right to create balanced highlights and soft shadows, "
            f"perfect reflections on pedestal surface. "
            f"Camera: tight product-focused depth of field, shot on Canon EOS R5 at 85mm, f/1.8 to f/2.8 (shallow DOF), "
            f"sharp focus on product, cinematic bokeh rendering background characters. "
            f"Styling: {visual_style}, brand tone: {brand_feel}. "
            f"{self.base_style}. Flawless post-production polish, color graded, high dynamic range, advertising quality."
        )

        print(f"📸 Commercial base prompt (truncated): {prompt[:160]}...")
        image_url = self._generate_single_image(prompt, seed=1100)

        if image_url:
            print("✅ Commercial reference image generated")
            base64_image = self._download_and_encode_image(image_url)
            if base64_image:
                self.base_character_image = base64_image
                return base64_image
            else:
                print("❌ Failed to download/encode commercial base image")
                return None
        else:
            print("❌ Failed to generate commercial reference image")
            return None

    # ---------- Scene generation (story/commercial) ----------
    def generate_scene_images(self, scenes: List[Dict], characters: Dict[str, Dict[str, str]], product_info: Dict[str, str] = None) -> List[List[str]]:
        """
        Generate two frames (start/end) per scene using the last generated image as a reference for consistency.
        Returns list of [start_url, end_url] lists (some entries may contain fewer items if generation failed).
        """
        if not self.base_character_image:
            print("❌ No base reference image (characters+product) found. Create it first.")
            return []

        all_scene_images: List[List[str]] = []
        total_images = len(scenes) * 2
        current_image_count = 0
        current_reference_image = self.base_character_image

        print(f"🎬 Generating scene images for {len(scenes)} scenes ({total_images} total frames expected)...")
        prompts = []
        for scene_idx, scene in enumerate(scenes, start=1):
            scene_number = scene.get('scene_number', scene_idx)
            title = scene.get('title', f"Scene {scene_number}")
            print(f"\n🎥 [{scene_idx}/{len(scenes)}] Scene {scene_number}: {title}")

            # Start frame (establishing)
            current_image_count += 1
            print(f"📸 [{current_image_count}/{total_images}] Generating start frame...")
            prompt = self._build_scene_prompt(scene, characters, product_info=product_info, frame_type="start")
            prompts.append(prompt)
            seed = 2000 + (scene_number * 7)
            image_url = self._generate(prompt, reference_image=current_reference_image, seed=seed)
            images = []
            if image_url:
                print("✅ Starting frame generated")
                current_reference_image = self._download_and_encode_image(image_url)
                images.append(current_reference_image)
            else:
                print("❌ Failed to generate starting frame")

            time.sleep(self.generation_delay)

            # # End frame (close-up/emotional)
            # current_image_count += 1
            # print(f"📸 [{current_image_count}/{total_images}] Generating end frame...")
            # end_prompt = self._build_scene_prompt(scene, characters, product_info=product_info, frame_type="end")
            # end_seed = seed + 42
            # end_image_url = self._generate_single_image(end_prompt, reference_image=current_reference_image, seed=end_seed)
            # if end_image_url:
            #     print("✅ Ending frame generated")
            #     current_reference_image = self._download_and_encode_image(end_image_url) or current_reference_image
            # else:
            #     print("❌ Failed to generate ending frame")
            
            scene_prompts = prompts
            scene_images = [u for u in images if u]
            all_scene_images.append(scene_images)

            # small delay between scenes to avoid API throttling
            if scene_idx < len(scenes):
                time.sleep(self.generation_delay)

        successful_images = sum(len(s) for s in all_scene_images)
        print(f"\n🎉 Scene generation complete: {successful_images}/{total_images} frames generated")
        return scene_prompts,all_scene_images

    # ---------- Prompt builder for scene frames ----------
    def _build_scene_prompt(self, scene: Dict, characters: Dict[str, Dict[str, str]], product_info: Dict[str, str] = None, frame_type: str = "start") -> str:
        """
        Build a professional photographic prompt for a single scene frame.
        frame_type: 'start' (establishing wide) or 'end' (close-up/emotional).
        Incorporates story events, character interactions, and product presence.
        """
        title = scene.get('title', '')
        story = scene.get('story_context').strip()
        script = scene.get('script').strip()
        print(f"Story context: {story}")
        actors = scene.get('actors', [])
        environment_description = scene.get('environment', scene.get('setting', 'neutral background'))
        lighting_hint = scene.get('lighting', 'consistent, cinematic lighting')
    
        # Guarantee we have a story to work from
        if not story:
            story = f"A scene titled '{title}' featuring {', '.join(actors)} in {environment_description}."
    
        # --- Consistency rule ---
        consistency_prompt = (
            "MAINTAIN EXACT same faces, hair, clothing, and product design as in the provided reference image. "
            "Do not alter character appearance or product design."
        )
    
        # --- Build character detail strings ---
        character_focus = []
        for actor in actors:
            char_data = self._find_character_data(actor, characters)
            if char_data:
                name = char_data.get('name', actor).strip('*').strip()
                desc_parts = []
                if char_data.get('hair'):
                    desc_parts.append(f"{char_data['hair']} hair")
                if char_data.get('distinctive'):
                    desc_parts.append(char_data['distinctive'])
                if char_data.get('outfit'):
                    desc_parts.append(f"wearing {char_data['outfit']}")
                character_focus.append(f"{name} ({', '.join(desc_parts)})" if desc_parts else name)
            else:
                if self.locked_character_descriptions:
                    character_focus.append("characters from reference image (maintain exact appearance)")
                else:
                    character_focus.append(actor)
    
        # --- Frame-specific composition ---
        if frame_type == "start":
            camera_style = "establishing wide-angle shot, showing full bodies and environment, dynamic composition, camera: 35mm, aperture f/4"
        elif frame_type == "end":
            camera_style = "tight close-up or action crop, focusing on faces, expressions, and hands, shallow depth of field, camera: 85mm, aperture f/2"
        else:
            camera_style = "balanced medium shot, cinematic framing"
    
        # --- Product presence ---
        product_block = ""
        if product_info:
            pname = product_info.get('name', 'the product')
            ptype = product_info.get('type', '')
            features = product_info.get('features', '')
            brand_feel = product_info.get('brand feel', '')
            visual_style = product_info.get('visual style', '')
            product_block = (
                f"\nInclude {ptype} '{pname}' prominently as described in the story, with identical design per reference image. "
                f"Key features: {features}. Brand feel: {brand_feel}. Visual style cues: {visual_style}."
            )
    
        # --- Final professional prompt ---
        prompt = (
            f"Scene: {title} — {story}\n\n"
            f"Script excerpt: {script}\n\n"
            f"Subjects: {', '.join(character_focus)} interacting naturally as described.\n"
            f"{consistency_prompt}{product_block}\n\n"
            f"Shot style: {camera_style}, {lighting_hint}, {self.base_style}. "
            f"Professional advertising photography, ultra high detail, cinematic composition, realistic skin textures, "
            f"natural expressions, accurate anatomy, HDR lighting, no distortions."
        )
        return prompt

    # ---------- Helper: find character data ----------
    def _find_character_data(self, actor: str, characters: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """Attempt various strategies to find character metadata for an actor name."""
        if not characters:
            return {}

        # direct key
        if actor in characters:
            return characters[actor]

        # check 'name' field in characters
        for _, data in characters.items():
            name_field = data.get('name', '')
            if name_field and (actor == name_field or actor.lower() in name_field.lower() or name_field.lower() in actor.lower()):
                return data

        # partial key match
        for key, data in characters.items():
            if actor.lower() in key.lower() or key.lower() in actor.lower():
                return data

        return {}

    # ---------- Product-focused image generator ----------
    def generate_product_images(self, product_info: Dict[str, str], reference_image: str = None) -> List[str]:
        """
        Generate a selection of product-centric images intended for adverts.
        Returns list of result URLs (not base64) for convenience.
        """
        product_name = product_info.get('name', 'the product')
        product_type = product_info.get('type', 'product')
        features = product_info.get('features', '')
        visual_style = product_info.get('visual style', 'modern, sleek')
        brand_feel = product_info.get('brand feel', 'premium, professional')

        print(f"🛍️ Generating product images for: {product_name}")

        prompts = [
            # Hero product with characters composed in background
            f"Hero product photography: {product_name}, a {product_type}, on a glossy black acrylic pedestal, "
            f"perfect reflections, hero light from above, shallow depth of field, characters from reference image softly in background. "
            f"{self.base_style}, {visual_style}, {brand_feel}.",

            # Lifestyle usage shot
            f"Lifestyle shot: characters from reference image actively using {product_name}, features visible: {features}, "
            f"natural environment, medium depth of field, cinematic color grade, {visual_style}, {brand_feel}.",

            # Close-up detail shot
            f"Macro close-up of {product_name} showing material and texture, extreme detail, studio lighting to reveal finishes, "
            f"{self.base_style}, product-centric composition, high dynamic range.",

            # Emotional context with product
            f"Emotive advert: smiling characters from reference image surrounding {product_name}, product in center, "
            f"warm lighting, lifestyle composition, {visual_style}, {brand_feel}."
        ]

        # If reference image provided, ensure character consistency
        product_images = []
        for i, p in enumerate(prompts, start=1):
            print(f"📸 Generating product image [{i}/{len(prompts)}]...")
            seed = 4000 + i * 13
            ref = reference_image if "characters from reference" in p else None
            image_url = self._generate_single_image(p, reference_image=ref, seed=seed)
            if image_url:
                product_images.append(image_url)
                print(f"✅ Product image {i} generated: {image_url}")
            else:
                print(f"❌ Product image {i} generation failed")
            time.sleep(self.generation_delay)

        return product_images

    # ---------- Streamlined commercial flow ----------
    def generate_commercial_sequence(self, scenes: List[Dict], characters: Dict[str, Dict[str, str]], product_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Streamlined commercial generation:
          1) Create combined base (characters + product)
          2) Generate scene images with product integrated
          3) Generate optional closing commercial frames (title/CALLOUT)
        """
        print("🎬 Starting commercial sequence...")

        result = {
            "base_character_image": None,
            "scene_prompt": [],
            "scene_images": [],
            "product_images": [],
            "commercial_frame_prompts":[],
            "commercial_frames": [],
            "success": False,
            "total_images": 0,
        }

        if not characters or not product_info:
            print("❌ Missing characters or product_info for commercial sequence")
            return result

        # Step 1: combined base
        result["base_character_image"] = self.create_base_commercial_reference_image(characters, product_info)
        if not result["base_character_image"]:
            print("❌ Failed to create commercial base - aborting sequence")
            return result

        time.sleep(self.generation_delay)

        # Step 2: narrative scenes with product present
        if scenes:
            result["scene_prompt"],result["scene_images"] = self.generate_scene_and_product_images(scenes, characters, product_info=product_info)


        # commercial closing frames (title card, CTA)
        result['commercial_frame_prompts'],result["commercial_frames"] = self._generate_commercial_frames(product_info, reference_image=result["base_character_image"])

        total_images = sum(len(x) for x in (result["scene_images"] + [result["commercial_frames"], result["product_images"]]))
        result["total_images"] = total_images
        result["success"] = total_images > 0

        print(f"\n🎉 Commercial sequence finished: {total_images} images generated (success={result['success']})")
        return result
    
    
    def generate_scene_and_product_images(self, scenes: List[Dict], characters: Dict[str, Dict[str, str]], product_info: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generate for each scene:
          - 2 narrative frames (start, end) with characters + product present
          - 1 product-focused advert frame with the same reference
        Returns a list of dicts: [{ "scene_number": int, "title": str, "scene_frames": [...], "product_frame": str }]
        """
        if not self.base_character_image:
            print("❌ No base reference image found. Create it first.")
            return []

        all_results = []
        current_reference_image = self.base_character_image

        print(f"🎬 Generating scene + product images for {len(scenes)} scenes...")

        for scene_idx, scene in enumerate(scenes, start=1):
            scene_number = scene.get('scene_number', scene_idx)
            title = scene.get('title', f"Scene {scene_number}")
            print(f"\n🎥 Scene {scene_number}: {title}")
            
            print("story of the scene is :\t",scene,'\n')

            # === Narrative start frame ===
            prompt = self._build_scene_prompt(scene, characters, product_info=product_info, frame_type="start")
            seed = 2000 + (scene_number * 7)
            url = self._generate_single_image(prompt, reference_image=current_reference_image, seed=seed)
            images = []
            if url:
                print("✅ Start frame generated")
                current_reference_image = self._download_and_encode_image(url) 
                images.append(current_reference_image)
            else:
                print("❌ Failed to generate start frame")

            time.sleep(self.generation_delay)

            # === Narrative end frame ===
            # end_prompt = self._build_scene_prompt(scene, characters, product_info=product_info, frame_type="end")
            # end_seed = seed + 42
            # end_url = self._generate_single_image(end_prompt, reference_image=current_reference_image, seed=end_seed)
            # if end_url:
            #     print("✅ End frame generated")
            #     current_reference_image = self._download_and_encode_image(end_url) or current_reference_image
            # else:
            #     print("❌ Failed to generate end frame")

            # time.sleep(self.generation_delay)

            # === Product-focused advert frame for this scene ===
            product_prompt = (
                f"Hero product photography for scene '{title}': {product_info.get('name', 'the product')}, "
                f"a {product_info.get('type', 'product')}, on glossy pedestal, perfect reflections, "
                f"characters from reference image in background, shallow depth of field, "
                f"{product_info.get('visual style', 'modern, sleek')}, {product_info.get('brand feel', 'premium, professional')}, "
                f"{self.base_style}"
            )
            prod_seed = 4000 + scene_number
            product_url = self._generate_single_image(product_prompt, reference_image=self.base_character_image, seed=prod_seed)
            if product_url:
                print("✅ Product advert frame generated")
                # I want base64 encoded product image
                product_image = self._download_and_encode_image(product_url)
            else:
                print("❌ Failed to generate product advert frame")

            all_results.append({
                "scene_number": scene_number,
                "scene_prompt": {"prompt": prompt, 
                                "product_prompt": prompt},
                "title": title,
                "scene_frames": [u for u in images if u],
                "product_frame_url": product_url,
                "product_frame_image": product_image if product_url else None
            })

        print(f"\n🎉 Finished generating {len(all_results)} scene+product sets")
        return all_results


    def _generate_commercial_frames(self, product_info: Dict[str, str], reference_image: str = None) -> List[str]:
        """
        Create title card / CTA frames for commercial use.
        Returns list of image URLs.
        """
        product_name = product_info.get('name', 'the product')
        brand_feel = product_info.get('brand feel', 'premium, professional')
        visual_style = product_info.get('visual style', 'modern')

        prompts = [
            f"Commercial title card for {product_name}. Elegant typography, logo placement, brand colors, minimalist layout, {brand_feel}, {visual_style}, high-end print quality.",
            f"Call-to-action frame for {product_name}: 'Available Now' overlay, product in subtle background, space for legal text and contact, {brand_feel}, commercial layout."
        ]

        frames = []
        for i, p in enumerate(prompts, start=1):
            print(f"📢 Generating commercial frame {i}/{len(prompts)}...")
            seed = 5000 + i * 11
            ref = reference_image if "characters from reference" in p else None
            url = self._generate_single_image(p, reference_image=ref, seed=seed, aspect_ratio="16:9")
            if url:
                frames.append(url)
                print(f"✅ Commercial frame {i} generated: {url}")
            else:
                print(f"❌ Commercial frame {i} failed")
            time.sleep(self.generation_delay)
        return prompts,frames