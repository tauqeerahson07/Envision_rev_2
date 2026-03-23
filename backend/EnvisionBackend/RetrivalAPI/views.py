from django.shortcuts import render
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.response import Response
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.response import Response
from typing import Dict, Any, List, Optional
from rest_framework.decorators import api_view, permission_classes,authentication_classes
from rest_framework import status
import json
import re
from .Services.checkpoints import checkpointer
from . import models, serializers
from .Services.script_generator import detect_project_type
from .main import build_workflow
from dotenv import load_dotenv
from .models import WorkflowCheckpoint
import os
import pprint
from moviepy import VideoFileClip, concatenate_videoclips
import tempfile
import base64
load_dotenv()

# Create your views here.
@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_project_and_scenes(request):
    """
    Get details of a particular project and its scenes for the authenticated user.
    Expects: { "project_id": ... }
    """
    try:
        data = json.loads(request.body)
        project_id = data.get('project_id')
        if not project_id:
            return Response({
                "status": "error",
                "message": "project_id is required."
            }, status=status.HTTP_400_BAD_REQUEST)

        project = models.Project.objects.get(id=project_id, user=request.user)
        project_serializer = serializers.ProjectSerializer(project)
        return Response({
            "status": "success",
            "project": project_serializer.data,
        })
    except models.Project.DoesNotExist:
        return Response({
            "status": "error",
            "message": "Project not found."
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def listProjects(request):
    projects = models.Project.objects.filter(user=request.user)
    data = [
        {
            "project_id":project.id,
            "project_name": project.title,
            "project_type": project.project_type
        }
        for project in projects
    ]
    return Response(data)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])  
@permission_classes([IsAuthenticated]) 
def CreateProject(request):
    """Initialize or continue project workflow"""
    try:
        data = json.loads(request.body)
        concept = data.get('concept', '').strip()
        num_of_scenes = data.get('num_scenes', 5)
        creativity = data.get('creativity', 'balanced').lower()
        project_title =f"Project on - {concept}"

        if not concept:
            return Response({
                "status": "error",
                "message": "Concept is required.",
                "error_code": "concept_required"
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            num_of_scenes = int(num_of_scenes)
            if num_of_scenes < 1 or num_of_scenes > 7:
                num_of_scenes = 5
        except (ValueError, TypeError):
            num_of_scenes = 5

        valid_creativity = ["factual", "creative", "balanced"]
        if creativity not in valid_creativity:
            creativity = "balanced"

        # Try to get existing project for this user and concept/title
        project, created = models.Project.objects.get_or_create(
            user=request.user,
            concept=concept,
            defaults={
                "num_scenes": num_of_scenes,
                "creativity_level": creativity,
                "title": project_title,
                "project_type": detect_project_type(concept)
            }
        )
        if not created:
            # Overwrite fields if project already exists
            project.num_scenes = num_of_scenes
            project.creativity_level = creativity
            project.title = project_title
            project.save()
            # Delete old scenes
            project.scenes.all().delete()

        init_state = {
            "concept": concept,
            "num_scenes": num_of_scenes,
            "creativity": creativity,
            "script": "",
            "characters": [],
            "product": {},
            "scenes": [],
            "project_title": project_title,
            "project_type": detect_project_type(concept)
        }

        app = build_workflow()
        thread_id = f"user-{request.user.id}-{project.id}"  
        config = {"configurable": {"thread_id": thread_id}} 
        state_after_script = app.invoke(init_state, config=config, interrupt_before="decide_rewrite")
        if state_after_script is None:
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Workflow did not return any state. Please check your workflow logic.",
                "error_code": "workflow_no_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        # Create scenes with custom titles if provided
        for scene_data in state_after_script.get("scenes", []):
            models.Scene.objects.get_or_create(
                project=project,
                project_type=project.project_type,
                scene_number=scene_data.get("scene_number", 1),
                script=scene_data.get("script"),
                characters=state_after_script.get("characters", []),
                product=state_after_script.get("product"),
                story_context=scene_data.get("story_context"),
                title=scene_data.get("title", f"Scene {scene_data.get('scene_number', 1)}"),
                environment=scene_data.get("environment", ""), 
                lighting=scene_data.get("lighting", ""),
            )

        serializer = serializers.ProjectSerializer(project)
        
        WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
        return Response({
            "status": "success",
            "message": "Script generated successfully.",
            "data": {
                "project": serializer.data,
            },
            "next_step": "review_script",
            "available_actions": ['accept_script', 'edit_scene', 'review_scene']
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
        return Response({
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "error_code": "internal_error"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])  
@permission_classes([IsAuthenticated])
def ReviewScript(request):
    """
    View details of a specific scene (read-only).
    Expects: { "project_id": ..., "scene_number": ... }
    """
    try:
        data = json.loads(request.body)
        project_id = data.get('project_id')
        scene_number = data.get('scene_number')

        if not project_id or not scene_number:
            return Response({
                "status": "error",
                "message": "Project ID and scene number are required.",
                "error_code": "missing_fields"
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            project = models.Project.objects.get(id=project_id, user=request.user)
        except models.Project.DoesNotExist:
            return Response({
                "status": "error",
                "message": "Project not found.",
                "error_code": "project_not_found"
            }, status=status.HTTP_404_NOT_FOUND)

        try:
            scene = models.Scene.objects.get(project=project, scene_number=scene_number)
        except models.Scene.DoesNotExist:
            return Response({
                "status": "error",
                "message": f"Scene {scene_number} not found.",
                "error_code": "scene_not_found"
            }, status=status.HTTP_404_NOT_FOUND)

        scene_serializer = serializers.SceneSerializer(scene)
        return Response({
            "status": "success",
            "message": f"Scene {scene_number} ({scene.title}) details.",
            "data": {
                "scene": scene_serializer.data,
                "scene_title": scene.title
            },
            "next_step": "review_script",
            "available_actions": ["accept_script", "edit_scene"]
        })

    except Exception as e:
        return Response({
            "status": "error",
            "message": "Internal server error.",
            "error_code": "internal_error"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def EditScene(request):
    try:
        data = json.loads(request.body)
        project_id = data.get('project_id')
        scene_number = int(data.get('scene_number'))
        edit_instructions = data.get('edit_instructions').strip()


        if not project_id or not scene_number or not edit_instructions:
            return Response({
                "status": "error",
                "message": "project_id, scene_number, and edit_instructions are required.",
                "error_code": "missing_fields"
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get project and scene
        try:
            project = models.Project.objects.get(id=project_id, user=request.user)
        except models.Project.DoesNotExist:
            return Response({
                "status": "error",
                "message": "Project not found.",
                "error_code": "project_not_found"
            }, status=status.HTTP_404_NOT_FOUND)

        try:
            scene_to_edit = models.Scene.objects.get(project=project, scene_number=scene_number)
        except models.Scene.DoesNotExist:
            return Response({
                "status": "error",
                "message": f"Scene {scene_number} not found.",
                "error_code": "scene_not_found"
            }, status=status.HTTP_404_NOT_FOUND)

        # Prepare state for scene rewriting
        thread_id = f"user-{request.user.id}-{project.id}"
        checkpoint_wrapper = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if checkpoint_wrapper is None:
            # Fallback: reconstruct state from DB
            existing_scenes = []
            for scene in project.scenes.all():
                existing_scenes.append({
                    'id': str(scene.id),
                    'scene_number': scene.scene_number,
                    'script': scene.script,
                    'story_context': scene.story_context,
                    'story': scene.story_context,
                    'title': scene.title
                })
            checkpoint_state = {
                "concept": project.concept,
                "num_scenes": project.num_scenes,
                "creativity": project.creativity_level,
                "scenes": existing_scenes,
                "project_title": project.title,
                "project_type": project.project_type,
            }
        else:
            checkpoint_state = checkpoint_wrapper.checkpoint

        # --- Always overwrite these fields for edit ---
        checkpoint_state["scene_to_edit"] = int(scene_number)
        checkpoint_state["needs_rewrite"] = True
        checkpoint_state["rewrite_instructions"] = edit_instructions
        checkpoint_state["rewrite_decision"] = "edit"

        # --- Unwrap channel_values if present ---
        if "channel_values" in checkpoint_state and "__root__" in checkpoint_state["channel_values"]:
            checkpoint_state = checkpoint_state["channel_values"]["__root__"]

        print("checkpoint_state before workflow invoke:")
        pprint.pprint(checkpoint_state)

        # --- Resume graph ---
        app = build_workflow(entry_point="rewrite_scene")
        config = {"configurable": {"thread_id": thread_id}}
        updated_state = app.invoke(checkpoint_state, config=config)

        print("DEBUG: updated_state after workflow:", updated_state)

        if isinstance(updated_state, dict) and updated_state.get("error"):
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": f"Workflow error: {updated_state['error']}",
                "error_code": "workflow_node_error"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if updated_state is None:
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Workflow did not return any state. Please check your workflow logic.",
                "error_code": "workflow_no_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Defensive: ensure updated_state is a dict
        if not isinstance(updated_state, dict):
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Workflow returned invalid state type.",
                "error_code": "invalid_workflow_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        updated_scenes = updated_state.get('scenes', [])

        scene_updated = False
        scene_to_edit = None

        for updated_scene in updated_scenes:
            scene_number_db = updated_scene.get('scene_number')
            try:
                db_scene = models.Scene.objects.get(project=project, scene_number=scene_number_db)
                db_scene.script = updated_scene.get('script', db_scene.script)
                db_scene.story_context = updated_scene.get('story_context', updated_scene.get('story', db_scene.story_context))
                db_scene.title = updated_scene.get('title', db_scene.title)
                db_scene.characters = updated_state.get('characters', db_scene.characters)
                db_scene.product = updated_state.get('product', db_scene.product)
                db_scene.save()

                if scene_number_db == scene_number:
                    scene_to_edit = db_scene
                    scene_updated = True
            except models.Scene.DoesNotExist:
                continue

        if not scene_updated or scene_to_edit is None:
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Failed to update scene - scene not found in response.",
                "error_code": "scene_update_failed"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        base_title = project.title.split(' - Scene')[0]
        project.title = f"{base_title} - Scene {scene_number} Updated"
        project.save()

        project.refresh_from_db()
        scene_serializer = serializers.SceneSerializer(scene_to_edit)
        
        WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
        
        return Response({
            "status": "success",
            "message": f"Scene {scene_number} ({scene_to_edit.title}) updated successfully.",
            "data": {
                "updated_scene": scene_serializer.data,
                "edit_instructions_used": edit_instructions
            },
            "next_step": "review_script",
            "available_actions": ["accept_script", "edit_scene"]
        })

    except Exception as e:
        import traceback
        print("Exception in EditScene:", traceback.format_exc())
        return Response({
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "trace": traceback.format_exc(),
            "error_code": "internal_error"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def EditAllScenes(request):
    """
    Edit all scenes in a project coherently while maintaining context.
    Expects: { "project_id": "...", "edit_instructions": "..." }
    """
    try:
        import pprint
        data = json.loads(request.body)
        project_id = data.get('project_id')
        edit_instructions = data.get('edit_instructions', '').strip()  

        if not project_id or not edit_instructions:
            return Response({
                "status": "error",
                "message": "project_id and edit_instructions are required.",
                "error_code": "missing_fields"
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get project
        try:
            project = models.Project.objects.get(id=project_id, user=request.user)
        except models.Project.DoesNotExist:
            return Response({
                "status": "error",
                "message": "Project not found.",
                "error_code": "project_not_found"
            }, status=status.HTTP_404_NOT_FOUND)

        # Check if project has scenes
        if not project.scenes.exists():
            return Response({
                "status": "error",
                "message": "No scenes found in this project.",
                "error_code": "no_scenes_found"
            }, status=status.HTTP_400_BAD_REQUEST)

        # Prepare state for all scenes rewriting
        thread_id = f"user-{request.user.id}-{project.id}"
        checkpoint_wrapper = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        
        if checkpoint_wrapper is None:
            # Fallback: reconstruct state from DB
            existing_scenes = []
            for scene in project.scenes.all().order_by('scene_number'):
                existing_scenes.append({
                    'scene_number': scene.scene_number,
                    'script': scene.script,
                    'story_context': scene.story_context,
                    'story': scene.story_context or scene.script,
                    'title': scene.title
                })
            checkpoint_state = {
                "concept": project.concept,
                "num_scenes": project.num_scenes,
                "creativity": project.creativity_level,
                "scenes": existing_scenes,
                "project_title": project.title,
                "project_type": project.project_type,
                "trigger_word": getattr(project, 'trigger_word', '')
            }
        else:
            checkpoint_state = checkpoint_wrapper.checkpoint

        # --- Set up for editing all scenes ---
        checkpoint_state["needs_rewrite"] = True
        checkpoint_state["rewrite_instructions"] = edit_instructions
        checkpoint_state["rewrite_decision"] = "edit"
        checkpoint_state["edit_all_scenes"] = True  
        
        # Remove specific scene editing fields if they exist
        checkpoint_state.pop("scene_to_edit", None)

        # --- Unwrap channel_values if present ---
        if "channel_values" in checkpoint_state and "__root__" in checkpoint_state["channel_values"]:
            checkpoint_state = checkpoint_state["channel_values"]["__root__"]

        print("checkpoint_state before workflow invoke (edit all scenes):")
        pprint.pprint(checkpoint_state)

        # --- Resume graph for all scenes ---
        app = build_workflow(entry_point="rewrite_scene")
        config = {"configurable": {"thread_id": thread_id}}
        updated_state = app.invoke(checkpoint_state, config=config)

        print("DEBUG: updated_state after workflow (all scenes):")
        pprint.pprint(updated_state)

        if isinstance(updated_state, dict) and updated_state.get("error"):
            return Response({
                "status": "error",
                "message": f"Workflow error: {updated_state['error']}",
                "error_code": "workflow_node_error"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if updated_state is None:
            return Response({
                "status": "error",
                "message": "Workflow did not return any state. Please check your workflow logic.",
                "error_code": "workflow_no_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Defensive: ensure updated_state is a dict
        if not isinstance(updated_state, dict):
            return Response({
                "status": "error",
                "message": "Workflow returned invalid state type.",
                "error_code": "invalid_workflow_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        updated_scenes = updated_state.get('scenes', [])
        print(f"DEBUG: Found {len(updated_scenes)} scenes in updated_state")
        
        # Check if scenes were actually modified by comparing content
        scenes_actually_changed = False
        scenes_updated_count = 0
        updated_scene_data = []

        # Update all scenes in the database
        for updated_scene in updated_scenes:
            scene_number_db = updated_scene.get('scene_number')
            try:
                db_scene = models.Scene.objects.get(project=project, scene_number=scene_number_db)
                new_story = updated_scene.get('story')
                new_script = updated_scene.get('script') or new_story
                new_context = updated_scene.get('story_context') or new_story or new_script
                scene_title = updated_scene.get('title', db_scene.title)

                # Check if content actually changed
                old_script = db_scene.script
                if new_script and new_script != old_script:
                    scenes_actually_changed = True
                    # print(f"DEBUG: Scene {scene_number_db} content changed")
                    # print(f"Old: {old_script[:100]}...")
                    # print(f"New: {new_script[:100]}...")
                    
                    # Apply character placeholder enforcement and update
                    db_scene.script = new_script
                    db_scene.story_context = new_context or new_script
                    db_scene.title = scene_title or db_scene.title
                    db_scene.save()
                    scenes_updated_count += 1
                else:
                    print(f"DEBUG: Scene {scene_number_db} content unchanged")
                
                # Serialize the scene for response (whether changed or not)
                scene_serializer = serializers.SceneSerializer(db_scene)
                updated_scene_data.append(scene_serializer.data)
                
            except models.Scene.DoesNotExist:
                print(f"DEBUG: Scene {scene_number_db} not found in database")
                continue

        # Check if any scenes were actually changed
        if not scenes_actually_changed:
            return Response({
                "status": "error", 
                "message": "Workflow completed but no scenes were actually modified. This might indicate an issue with the workflow logic or the edit instructions were not processed correctly.",
                "error_code": "no_actual_changes",
                "debug_info": {
                    "edit_instructions": edit_instructions,
                    "scenes_in_response": len(updated_scenes),
                    "workflow_state_keys": list(updated_state.keys()) if isinstance(updated_state, dict) else "not_dict",
                    "edit_all_scenes_flag": checkpoint_state.get("edit_all_scenes"),
                    "rewrite_decision": checkpoint_state.get("rewrite_decision")
                }
            }, status=status.HTTP_400_BAD_REQUEST)

        if scenes_updated_count == 0:
            return Response({
                "status": "error",
                "message": "No scenes were actually updated in the database.",
                "error_code": "scenes_update_failed",
                "debug_info": {
                    "scenes_found_in_response": len(updated_scenes),
                    "edit_instructions": edit_instructions
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Update project title to reflect the edit
        base_title = project.title.split(' - ')[0]  # Remove any existing suffixes
        project.title = f"{base_title} - All Scenes Updated"
        project.save()

        project.refresh_from_db()
        project_serializer = serializers.ProjectSerializer(project)
        
        WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
        
        return Response({
            "status": "success",
            "message": f"Successfully updated {scenes_updated_count} scenes coherently.",
            "data": {
                "project": project_serializer.data,
                "scenes_updated_count": scenes_updated_count,
                "total_scenes": project.scenes.count(),
                "edit_instructions_used": edit_instructions
            },
            "next_step": "review_script",
            "available_actions": ["accept_script", "edit_scene", "edit_all_scenes"]
        })

    except json.JSONDecodeError:
        return Response({
            "status": "error",
            "message": "Invalid JSON format in request body.",
            "error_code": "invalid_json"
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        import traceback
        print("Exception in EditAllScenes:", traceback.format_exc())
        return Response({
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "trace": traceback.format_exc(),
            "error_code": "internal_error"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def CreateImages(request):
    """API to generate and store images for all Scenes of a particular project"""
    try:
        data = json.loads(request.body)
        project_id = data.get('project_id')

        if not project_id:
            return Response({
                "status": "error",
                "message": "project_id is required.",
                "error_code": "missing_fields"
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            project = models.Project.objects.get(id=project_id, user=request.user)
        except models.Project.DoesNotExist:
            return Response({
                "status": "error",
                "message": "Project not found.",
                "error_code": "project_not_found"
            }, status=status.HTTP_404_NOT_FOUND)

        scenes = []
        all_characters = {}

        for scene in project.scenes.order_by('scene_number'):
            chars = scene.characters
            # Convert dict to list if needed
            if isinstance(chars, dict):
                for k, v in chars.items():
                    all_characters[k] = v
                chars = list(chars.values())
            elif isinstance(chars, list):
                for char in chars:
                    if isinstance(char, dict) and "name" in char:
                        all_characters[char["name"]] = char
            scenes.append({
                "scene_number": scene.scene_number,
                "title": scene.title,
                "script": scene.script,
                "story_context": scene.story_context,
                "image_prompt": scene.image_prompt,
                "characters": chars,
                "product": scene.product,
                "environment": scene.environment,  
                "lighting": scene.lighting,        
            })

        state = {
            "concept": project.concept,
            "characters": all_characters,  
            "scenes": scenes,
            "product": getattr(project, "product", {}),
            "project_type": project.project_type,
            "needs_image_edit":True,
        }
        thread_id = f"user-{request.user.id}-{project.id}"
        app = build_workflow(entry_point="generate_images")
        config = {"configurable": {"thread_id": thread_id}}
        updated_state = app.invoke(state, config=config)

        if not isinstance(updated_state, dict):
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Workflow returned invalid state type.",
                "error_code": "invalid_workflow_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        image_prompts = updated_state.get("image_prompts", [])
        scene_images = updated_state.get("scene_images", [])
        updated_scenes = []
        for idx, scene_image in enumerate(scene_images):
            try:
                scene_obj = project.scenes.get(scene_number=scenes[idx]["scene_number"])
                if isinstance(scene_image, list) and scene_image:
                    scene_obj.image = scene_image[0]
                else:
                    scene_obj.image = scene_image
                    
                if idx < len(image_prompts):
                    scene_obj.image_prompt = image_prompts[idx]
                scene_obj.save()
                updated_scenes.append({
                    "scene_number": scene_obj.scene_number,
                    "title": scene_obj.title,
                    "image": scene_obj.image,
                    "image_prompt": scene_obj.image_prompt
                    })
            except Exception:
                continue
        WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()    

        return Response({
            "status": "success",
            "message": f"Images generated and saved for {len(updated_scenes)} scenes.",
            "scenes": updated_scenes
        }, status=status.HTTP_200_OK)

    except Exception as e:
        import traceback

        return Response({
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "trace": traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])        
def EditImages(request):
    """API to edit images for a particular scene based on user instructions"""
    try:
        data = json.loads(request.body)
        project_id = data.get('project_id')
        scene_number = data.get('scene_number')
        edit_instructions = data.get('edit_instructions', '').strip()
        style = data.get('style', 'realistic').strip().lower()
        if not project_id or not scene_number or not edit_instructions:
            return Response({
                "status": "error",
                "message": "project_id, scene_number, and edit_instructions are required.",
                "error_code": "missing_fields"
            }, status=status.HTTP_400_BAD_REQUEST)
        try:
            project = models.Project.objects.get(id=project_id, user=request.user)
        except models.Project.DoesNotExist:
            return Response({
                "status": "error",
                "message": "Project not found.",
                "error_code": "project_not_found"
            }, status=status.HTTP_404_NOT_FOUND)
        try:
            scene = models.Scene.objects.get(project=project, scene_number=scene_number)
        except models.Scene.DoesNotExist:
            return Response({
                "status": "error",
                "message": f"Scene {scene_number} not found.",
                "error_code": "scene_not_found"
            }, status=status.HTTP_404_NOT_FOUND)
            
        # Prepare state for scene rewriting
        thread_id = f"user-{request.user.id}-{project.id}"
        checkpoint_wrapper = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if checkpoint_wrapper is None:
            # Fallback: reconstruct state from DB
            existing_scenes = []
            for scene in project.scenes.all():
                existing_scenes.append({
                    'id': str(scene.id),
                    'title': scene.title,
                    'scene_number': scene.scene_number,
                    'script': scene.script,
                    'story_context': scene.story_context,
                    'scene_image_prompt':scene.image_prompt,
                    'scene_image':scene.image
                })
            checkpoint_state = {
                "concept": project.concept,
                "num_scenes": project.num_scenes,
                "creativity": project.creativity_level,
                "scenes": existing_scenes,
                "project_title": project.title,
                "project_type": project.project_type,
            }
        else:
            checkpoint_state = checkpoint_wrapper.checkpoint
            
        # Step 1: Combine story contexts from all scenes
        combined_context = " ".join(
            scene.story_context for scene in project.scenes.all() if scene.story_context
        )

        # Step 2: Build per-scene edit instruction
        aggregated_instructions = (
            f"Edit the image in a {style} style. "
            f"Apply these modifications: {edit_instructions}. "
            f"Ensure the visual tone matches the overall story context: {combined_context}. "
            f"For this specific scene, focus on: {scene.story_context}."
        )

        # --- Always overwrite these fields for edit ---
        checkpoint_state["scene_to_edit"] = int(scene_number)
        checkpoint_state["style"] = style
        checkpoint_state["needs_rewrite"] = True
        checkpoint_state["rewrite_instructions"] = aggregated_instructions
        checkpoint_state["rewrite_decision"] = "edit"

        # --- Unwrap channel_values if present ---
        if "channel_values" in checkpoint_state and "__root__" in checkpoint_state["channel_values"]:
            checkpoint_state = checkpoint_state["channel_values"]["__root__"]

        print("checkpoint_state before workflow invoke:")
        pprint.pprint(checkpoint_state)

        # --- Resume graph ---
        app = build_workflow(entry_point="edit_images")
        config = {"configurable": {"thread_id": thread_id}}
        updated_state = app.invoke(checkpoint_state, config=config)

        print("DEBUG: updated_state after workflow:", updated_state)

        if isinstance(updated_state, dict) and updated_state.get("error"):
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": f"Workflow error: {updated_state['error']}",
                "error_code": "workflow_node_error"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if updated_state is None:
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Workflow did not return any state. Please check your workflow logic.",
                "error_code": "workflow_no_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Defensive: ensure updated_state is a dict
        if not isinstance(updated_state, dict):
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Workflow returned invalid state type.",
                "error_code": "invalid_workflow_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        updated_scenes = updated_state.get('scenes', [])

        scene_updated = False
        scene_to_edit = None

        for updated_scene in updated_scenes:
            scene_number_db = updated_scene.get('scene_number')
            try:
                db_scene = models.Scene.objects.get(project=project, scene_number=scene_number_db)
                db_scene.script = updated_scene.get('script', db_scene.script)
                db_scene.story_context = updated_scene.get('story_context', updated_scene.get('story', db_scene.story_context))
                db_scene.title = updated_scene.get('title', db_scene.title)
                db_scene.image_prompt = updated_scene.get('scene_image_prompt', db_scene.image_prompt)
                db_scene.image = updated_scene.get('scene_image', db_scene.image)
                db_scene.save()

                if scene_number_db == scene_number:
                    scene_to_edit = db_scene
                    scene_updated = True
            except models.Scene.DoesNotExist:
                continue

        if not scene_updated or scene_to_edit is None:
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Failed to update scene - scene not found in response.",
                "error_code": "scene_update_failed"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # base_title = project.title.split(' - Scene')[0]
        # project.title = f"{base_title} - Scene {scene_number} Updated"
        project.save()

        project.refresh_from_db()
        scene_serializer = serializers.SceneSerializer(scene_to_edit)
        
        WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
        
        return Response({
            "status": "success",
            "message": f"Scene {scene_number} ({scene_to_edit.title}) updated successfully.",
            "data": {
                "updated_scene": scene_serializer.data,
                "edit_instructions_used": edit_instructions
            },
            "next_step": "review_script",
            "available_actions": ["accept_script", "edit_scene"]
        })

    except Exception as e:
        import traceback
        print("Exception in EditScene:", traceback.format_exc())
        return Response({
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "trace": traceback.format_exc(),
            "error_code": "internal_error"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def EditAllImages(request):
    """API to edit images for all scenes based on user instructions"""
    try:
        data = json.loads(request.body)
        project_id = data.get('project_id')
        edit_instructions = data.get('edit_instructions', '').strip()
        style = data.get('style', 'realistic').strip().lower()
        if not project_id or not edit_instructions:
            return Response({
                "status": "error",
                "message": "project_id and edit_instructions are required.",
                "error_code": "missing_fields"
            }, status=status.HTTP_400_BAD_REQUEST)
        try:
            project = models.Project.objects.get(id=project_id, user=request.user)
        except models.Project.DoesNotExist:
            return Response({
                "status": "error",
                "message": "Project not found.",
                "error_code": "project_not_found"
            }, status=status.HTTP_404_NOT_FOUND)
        
        scenes = []
        all_characters = {}

        for scene in project.scenes.order_by('scene_number'):
            chars = scene.characters
            # Convert dict to list if needed
            if isinstance(chars, dict):
                for k, v in chars.items():
                    all_characters[k] = v
                chars = list(chars.values())
            elif isinstance(chars, list):
                for char in chars:
                    if isinstance(char, dict) and "name" in char:
                        all_characters[char["name"]] = char
            scenes.append({
                "id": str(scene.id),
                "scene_number": scene.scene_number,
                "title": scene.title,
                "script": scene.script,
                "story_context": scene.story_context,
                "image_prompt": scene.image_prompt,
                "scene_image": scene.image,
                "characters": chars,
                "product": scene.product,
                "environment": scene.environment,  
                "lighting": scene.lighting,        
            })
        
        combined_context = " ".join(scene.story_context for scene in project.scenes.all() if scene.story_context)

        aggregated_instructions = (
            f"Generate visuals in a {style} style. "
            f"Incorporate the following edits: {edit_instructions}. "
            f"Base the visuals on the overall story context: {combined_context}."
        )


        state = {
            "concept": project.concept,
            "characters": all_characters,  
            "scenes": scenes,
            "product": getattr(project, "product", {}),
            "project_type": project.project_type,
            "style": style,
            "needs_rewrite": True,
            "rewrite_instructions": aggregated_instructions,
            "rewrite_decision": "edit"
        }
        thread_id = f"user-{request.user.id}-{project.id}"
        app = build_workflow(entry_point="edit_images")
        config = {"configurable": {"thread_id": thread_id}}
        updated_state = app.invoke(state, config=config)
        if not isinstance(updated_state, dict):
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Workflow returned invalid state type.",
                "error_code": "invalid_workflow_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        image_prompts = updated_state.get("image_prompts", [])
        scene_images = updated_state.get("scene_images", [])
        updated_scenes = []
        for idx, scene_image in enumerate(scene_images):
            try:
                scene_obj = project.scenes.get(scene_number=scenes[idx]["scene_number"])
                if isinstance(scene_image, list) and scene_image:
                    scene_obj.image = scene_image[0]
                else:
                    scene_obj.image = scene_image
                    
                if idx < len(image_prompts):
                    scene_obj.image_prompt = image_prompts[idx]
                scene_obj.save()
                updated_scenes.append({
                    "scene_number": scene_obj.scene_number,
                    "title": scene_obj.title,
                    "image": scene_obj.image,
                    "image_prompt": scene_obj.image_prompt
                    })
            except Exception:
                continue
        WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
        return Response({
            "status": "success",
            "message": f"Images edited and saved for {len(updated_scenes)} scenes.",
            "scenes": updated_scenes
        }, status=status.HTTP_200_OK)
    except Exception as e:
        import traceback

        return Response({
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "trace": traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def GenerateVideo(request):
    """Generate and store video for all project scenes, and save final merged video."""
    try:
        data = json.loads(request.body)
        project_id = data.get('project_id')
        if not project_id:
            return Response({
                "status": "error",
                "message": "project_id is required.",
                "error_code": "missing_fields"
            }, status=status.HTTP_400_BAD_REQUEST)

        project = models.Project.objects.get(id=project_id, user=request.user)

        # Prepare workflow state
        state = {
            "scenes": [
                {
                    "scene_number": scene.scene_number,
                    "image": scene.image,
                    "prompt": scene.story_context,
                }
                for scene in project.scenes.order_by("scene_number")
                if scene.image and scene.story_context
            ]
        }

        thread_id = f"user-{request.user.id}-{project.id}"
        app = build_workflow(entry_point="generate_video")
        config = {"configurable": {"thread_id": thread_id}}
        updated_state = app.invoke(state, config=config)

        if not isinstance(updated_state, dict):
            WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()
            return Response({
                "status": "error",
                "message": "Workflow returned invalid state type.",
                "error_code": "invalid_workflow_state"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        video_data = updated_state.get("scene_videos")
        if not video_data:
            return Response({
                "status": "error",
                "message": "No video data returned from workflow.",
                "error_code": "no_video_data"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # --- Save each scene’s video ---
        for scene_video in video_data:
            scene_number = scene_video.get("scene_number")
            video_base64 = scene_video.get("video")
            if not video_base64:
                continue

            scene = project.scenes.filter(scene_number=scene_number).first()
            if scene:
                scene.video = video_base64
                scene.save()

        WorkflowCheckpoint.objects.filter(thread_id=thread_id).delete()

        # --- Combine videos ---
        scenes = project.scenes.order_by("scene_number")
        base64_videos = [scene.video for scene in scenes if scene.video]
        temp_files = []
        video_clips = []

        try:
            for base64_video in base64_videos:
                data = base64_video.split(",")[1] if "," in base64_video else base64_video
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file.write(base64.b64decode(data))
                temp_file.close()
                temp_files.append(temp_file.name)
                video_clips.append(VideoFileClip(temp_file.name))

            if not video_clips:
                return Response({
                    "status": "error",
                    "message": "No valid scene videos to merge.",
                    "error_code": "empty_merge"
                }, status=status.HTTP_400_BAD_REQUEST)

            final_clip = concatenate_videoclips(video_clips, method="compose")
            temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            final_clip.write_videofile(temp_output_file.name, codec="libx264", audio_codec="aac")

            with open(temp_output_file.name, "rb") as f:
                encoded_video = f"data:video/mp4;base64,{base64.b64encode(f.read()).decode('utf-8')}"

            project.video = encoded_video
            project.save()

            return Response({
                "status": "success",
                "message": "Scene videos generated and final video saved.",
                "video": encoded_video
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                "status": "error",
                "message": f"Error processing final video: {str(e)}",
                "error_code": "video_processing_error"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Cleanup temp files
            for clip in video_clips:
                clip.close()
            for path in temp_files:
                if os.path.exists(path):
                    os.remove(path)
            if 'temp_output_file' in locals() and os.path.exists(temp_output_file.name):
                os.remove(temp_output_file.name)

    except models.Project.DoesNotExist:
        return Response({
            "status": "error",
            "message": "Project not found.",
            "error_code": "project_not_found"
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
@authentication_classes([JWTAuthentication])  
@permission_classes([IsAuthenticated])
def GetProjectStatus(request, project_id):
    """Get current project status and next available actions"""
    try:
        project = models.Project.objects.get(id=project_id, user=request.user)
        serializer = serializers.ProjectSerializer(project)
        
        current_step = 'completed' if 'Completed' in project.title else 'review_script'
        
        # Define next actions based on current step
        next_actions = {
            'generating_script': ['wait'],
            'review_script': ['accept_script', 'edit_scene'],
            'edit_scene': ['provide_edit_instructions'],
            'completed': ['view_results']
        }
        
        return Response({
            "project": serializer.data,
            "current_step": current_step,
            "available_actions": next_actions.get(current_step, [])
        })
        
    except models.Project.DoesNotExist:
        return Response(
            {"error": "Project not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": f"Internal server error: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )