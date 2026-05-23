from django.db import models

# Create your models here.
from django.db import models

class ResearchReport(models.Model):
    """Stores each research job and its result."""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    topic = models.CharField(max_length=500)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # The final report content
    report_markdown = models.TextField(blank=True, null=True)
    report_pdf_path = models.CharField(max_length=500, blank=True, null=True)
    
    # Metadata from the pipeline
    sources_count = models.IntegerField(default=0)
    entities_count = models.IntegerField(default=0)
    critic_score = models.IntegerField(default=0)
    revision_count = models.IntegerField(default=0)
    
    error_message = models.TextField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.topic[:50]} — {self.status}"
    
    class Meta:
        ordering = ['-created_at']