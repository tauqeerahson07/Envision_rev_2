from django.urls import path
from . import views

urlpatterns = [
        # Project management endpoints
    path('list-projects/', views.listProjects, name='list_projects'),
    path('create-project/', views.CreateProject, name='create_project'),
    
    # Script review and editing endpoints
    path('review-script/', views.ReviewScript, name='review_script'),
    path('edit-scene/', views.EditScene, name='edit_scene'),
    path('edit-all-scenes/', views.EditAllScenes, name='edit_all_scenes'),
    
    # Project status endpoint
    path('project-status/<uuid:project_id>/', views.GetProjectStatus,name='project_status'),
    
    path('project/scenes/', views.get_project_and_scenes, name='get_project_and_scenes'),
    
    # Generate images endpoint
    path('generate-images/', views.CreateImages, name='generate_images'),
    
    path('edit-images/', views.EditImages, name='edit_images'),
    path('edit-all-images/', views.EditAllImages, name='edit_all_images'),
    
    # generate video endpoint
    path('generate-video/', views.GenerateVideo, name='generate_video'),
]