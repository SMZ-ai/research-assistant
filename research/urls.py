from django.urls import path
from . import views

app_name = 'research'

urlpatterns = [
    path('', views.index, name='index'),
    path('start/', views.start_research, name='start'),
    path('report/<int:report_id>/', views.view_report, name='report'),
]