"""
Illustration Agent — cleans up image placeholders in the draft.
Image generation is disabled (no OpenAI key).
Can be re-enabled later by adding DALL-E or Stable Diffusion.
"""
import re
from pathlib import Path

# Create media folder just in case other parts need it
Path('media/illustrations').mkdir(parents=True, exist_ok=True)


def run_illustration_agent(state: dict) -> dict:
    """
    Replaces IMAGE_PLACEHOLDER tags with clean figure captions.
    No actual image generation — keeps the report looking clean.
    """
    draft = state.get('draft', '')

    if not draft:
        state['illustrations'] = []
        state['status']        = 'illustration_complete'
        return state

    print("[Illustration Agent] Replacing image placeholders with captions")

    # Replace: ![Figure 1: Description](IMAGE_PLACEHOLDER_1)
    # With:    > 📊 *Figure 1: Description*
    cleaned_draft = re.sub(
        r'!\[([^\]]+)\]\(IMAGE_PLACEHOLDER_\d+\)',
        r'\n> 📊 *Figure: \1*\n',
        draft
    )

    # Also clean any remaining raw IMAGE_PLACEHOLDER text
    cleaned_draft = re.sub(
        r'IMAGE_PLACEHOLDER_\d+',
        '',
        cleaned_draft
    )

    state['draft']         = cleaned_draft
    state['illustrations'] = []
    state['status']        = 'illustration_complete'

    print("[Illustration Agent] Done")
    return state