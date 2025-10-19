from django.db import models
from django.contrib.auth.models import User,Group
import uuid

# Create your models here.

class Project(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='projects')
    title = models.CharField(max_length=255)
    concept = models.TextField()
    num_scenes = models.IntegerField()
    creativity_level = models.CharField(
        max_length=20,
        choices=[
            ('factual', 'Factual'), 
            ('creative', 'Creative'),   
            ('balanced', 'Balanced'), 
        ],
        default='balanced'
    )
    project_type = models.CharField(
        max_length=20,
        choices=[
            ('story', 'Story'),
            ('commercial', 'Commercial'),
        ],
        default='story'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    video = models.TextField(null=True, blank=True)  # Store video as base64

    def __str__(self):
        return self.title

class Scene(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='scenes')
    project_type = models.CharField(max_length=20, default='story')
    title = models.CharField(max_length=255)
    scene_number = models.IntegerField()
    script = models.TextField()
    characters = models.JSONField(default=list, blank=True)
    product = models.JSONField(default=dict, blank=True, null=True)
    environment = models.TextField(blank=True, default='')
    lighting = models.TextField(blank=True, default='')
    story_context = models.TextField(blank=True)
    image_prompt = models.TextField(blank=True)
    image = models.TextField(null=True, blank=True)  # Store image as base 64
    video = models.TextField(null=True, blank=True)  # Store video as base64
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['scene_number']
        unique_together = ['project', 'scene_number']

    def __str__(self):
        return f"{self.project.title} - Scene {self.scene_number}"
# class Images(models.Model):
#     project_id = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='images')
#     image_data = models.BinaryField() 
#     created_at = models.DateTimeField(auto_now_add=True)
    
# class Videos(models.Model):
#     project_id = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='videos')
#     video_url = models.BinaryField()
#     created_at = models.DateTimeField(auto_now_add=True)

# Workflow checkpoints
class WorkflowCheckpoint(models.Model):
    thread_id = models.TextField()
    version = models.IntegerField(default=1)
    state_json = models.JSONField()

    class Meta:
        unique_together = ("thread_id", "version") 
