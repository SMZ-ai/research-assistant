import os
import ast
from collections import Counter, defaultdict
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception as e:
    print(f"[NER Agent] SpaCy not available: {e}")
    SPACY_AVAILABLE = False


def map_spacy_label(label: str) -> str:
    mapping = {
        'PERSON': 'person', 'ORG': 'organization', 'GPE': 'location',
        'DATE': 'date', 'PRODUCT': 'technology', 'WORK_OF_ART': 'publication',
    }
    return mapping.get(label, 'other')


def extract_with_spacy(classified_sources: list) -> tuple:
    entity_counter = Counter()
    entity_to_sources = defaultdict(list)
    entity_to_label = {}
    if not SPACY_AVAILABLE:
        return entity_counter, entity_to_sources, entity_to_label
    for source in classified_sources:
        content = source.get('content', '')
        if not content:
            continue
        doc = nlp(content[:5000])
        for ent in doc.ents:
            if ent.label_ in ('PERSON', 'ORG', 'GPE', 'DATE', 'PRODUCT', 'WORK_OF_ART'):
                entity_text = ent.text.strip()
                if len(entity_text) < 2 or entity_text.isdigit():
                    continue
                entity_counter[entity_text] += 1
                entity_to_sources[entity_text].append(source.get('url', ''))
                entity_to_label[entity_text] = map_spacy_label(ent.label_)
    return entity_counter, entity_to_sources, entity_to_label


def extract_tech_entities(topic: str, sources: list) -> list:
    if not sources:
        return []
    combined_text = " ".join([s.get('content', '')[:500] for s in sources[:5]])
    if not combined_text.strip():
        return []
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY'),
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract named entities from research text.
            Focus on: AI models, ML frameworks, algorithms, companies, researchers, papers, datasets.
            Return ONLY a Python list of dicts, no markdown:
            [{"text": "GPT-4", "label": "technology", "frequency": 1}]
            Labels: technology, person, organization, publication, dataset, algorithm"""),
            ("human", "Research topic: {topic}\n\nText:\n{text}")
        ])
        chain = prompt | llm
        response = chain.invoke({"topic": topic, "text": combined_text})
        content = response.content.strip()
        if '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
            if content.startswith('python'):
                content = content[6:].strip()
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            content = content[start:end]
        entities = ast.literal_eval(content)
        if isinstance(entities, list):
            valid = []
            for e in entities:
                if isinstance(e, dict) and 'text' in e and 'label' in e:
                    valid.append({
                        'text': str(e['text']).strip(),
                        'label': str(e['label']).strip(),
                        'frequency': int(e.get('frequency', 1)),
                        'source_urls': [],
                    })
            return valid
    except Exception as e:
        print(f"[NER Agent] Groq extraction failed: {e}")
    return []


def run_ner_agent(state: dict) -> dict:
    classified_sources = state.get('classified_sources', [])
    if not classified_sources:
        state['entities'] = []
        state['status'] = 'ner_complete'
        return state

    print(f"[NER Agent] Running NER on {len(classified_sources)} sources")

    entity_counter, entity_to_sources, entity_to_label = extract_with_spacy(classified_sources)
    spacy_entities = []
    for entity_text, count in entity_counter.most_common(50):
        spacy_entities.append({
            'text': entity_text,
            'label': entity_to_label.get(entity_text, 'other'),
            'frequency': count,
            'source_urls': list(set(entity_to_sources[entity_text])),
        })
    print(f"[NER Agent] SpaCy found {len(spacy_entities)} entities")

    tech_entities = extract_tech_entities(state['topic'], classified_sources)
    print(f"[NER Agent] Groq found {len(tech_entities)} additional entities")

    seen_texts = {e['text'].lower() for e in spacy_entities}
    for tech_ent in tech_entities:
        if tech_ent['text'].lower() not in seen_texts:
            spacy_entities.append(tech_ent)
            seen_texts.add(tech_ent['text'].lower())

    all_entities = sorted(spacy_entities, key=lambda x: x.get('frequency', 0), reverse=True)
    print(f"[NER Agent] Total unique entities: {len(all_entities)}")

    state['entities'] = all_entities
    state['status'] = 'ner_complete'
    return state