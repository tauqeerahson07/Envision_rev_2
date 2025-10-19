from rest_framework import serializers
from .models import Project, Scene

class SceneSerializer(serializers.ModelSerializer):
    project_title = serializers.CharField(source='project.title', read_only=True)

    class Meta:
        model = Scene
        fields = ['id', 'scene_number', 'script', 'story_context', 'created_at', 'project_title', 'title','image_prompt','image','video',
                'characters', 'product', 'project_type', 'environment', 'lighting']

class ProjectSerializer(serializers.ModelSerializer):
    scenes = SceneSerializer(many=True, read_only=True)
    
    class Meta:
        model = Project
        fields = ['id','title', 'concept', 'num_scenes', 'creativity_level', 
                'created_at', 'updated_at', 'scenes']

class ProjectCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ['title', 'concept', 'num_scenes', 'creativity_level']