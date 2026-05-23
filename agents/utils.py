"""Utility functions used across agents."""
import os
import markdown2
from pathlib import Path

def save_pdf(markdown_content: str, report_id: int) -> str:
    """
    Converts Markdown report to PDF and saves it.
    Returns the relative path to the saved PDF.
    """
    if not markdown_content:
        return ""
    
    try:
        import weasyprint
        
        # Convert Markdown to HTML
        html_content = markdown2.markdown(
            markdown_content,
            extras=['fenced-code-blocks', 'tables', 'strike']
        )
        
        # Add basic styling
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Georgia, serif; max-width: 800px; margin: 0 auto; padding: 40px; }}
                h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px; }}
                h2 {{ color: #16213e; }}
                img {{ max-width: 100%; }}
                code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
            </style>
        </head>
        <body>{html_content}</body>
        </html>
        """
        
        # Save PDF
        output_dir = Path('media/reports')
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"report_{report_id}.pdf"
        
        weasyprint.HTML(string=full_html).write_pdf(str(pdf_path))
        
        return str(pdf_path).replace('media/', '')  # relative path
        
    except Exception as e:
        print(f"PDF generation failed: {e}")
        return ""