from django import forms

class ResearchForm(forms.Form):
    topic = forms.CharField(
        max_length=500,
        label="Research Topic",
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g. The impact of Mixture of Experts on LLM efficiency',
            'class': 'form-control form-control-lg',
        })
    )
