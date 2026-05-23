from django.shortcuts import render

# Create your views here.
import threading
import markdown2
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from .models import ResearchReport
from .forms import ResearchForm
from agents.orchestrator import run_research_pipeline


def index(request):
    """Home page with form and list of past reports."""
    form = ResearchForm()
    reports = ResearchReport.objects.all()[:10]
    return render(request, 'research/index.html', {
        'form': form,
        'reports': reports,
    })


def start_research(request):
    """Handles form submission — creates a report and starts the pipeline."""
    if request.method != 'POST':
        return redirect('research:index')
    
    form = ResearchForm(request.POST)
    if not form.is_valid():
        return redirect('research:index')
    
    topic = form.cleaned_data['topic']
    
    # Create the database record immediately
    report = ResearchReport.objects.create(
        topic=topic,
        status='running',
    )
    
    # Run the pipeline in a background thread so the user isn't waiting
    def run_pipeline():
        try:
            result = run_research_pipeline(topic)
            
            report.status = 'completed'
            report.report_markdown = result.get('final_report', '')
            report.sources_count = len(result.get('raw_sources', []))
            report.entities_count = len(result.get('entities', []))
            report.critic_score = result.get('critic_score', 0)
            report.revision_count = result.get('revision_count', 0)
            report.completed_at = timezone.now()
            
            # Generate PDF
            from agents.utils import save_pdf
            pdf_path = save_pdf(result.get('final_report', ''), report.id)
            report.report_pdf_path = pdf_path
            
        except Exception as e:
            report.status = 'failed'
            report.error_message = str(e)
        
        report.save()
    
    thread = threading.Thread(target=run_pipeline)
    thread.daemon = True
    thread.start()
    
    return redirect('research:report', report_id=report.id)


def view_report(request, report_id):
    """Displays a single report."""
    report = get_object_or_404(ResearchReport, id=report_id)
    
    # Convert Markdown to HTML for display
    report_html = ''
    if report.report_markdown:
        report_html = markdown2.markdown(
            report.report_markdown,
            extras=['fenced-code-blocks', 'tables', 'strike']
        )
    
    return render(request, 'research/report.html', {
        'report': report,
        'report_html': report_html,
    })