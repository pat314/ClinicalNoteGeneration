import pandas as pd
import json
import logging
import inspect

from summarize_all import RetrievalResult, summarize, question_extract_then_fill, naive_extract_after_terms
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger()


def note_complete(df: pd.DataFrame, dialogue_column: str, index_column: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Generates complete notes from retrieval results.
    """

    df = df.rename(columns={index_column: 'encounter_id'})

    notes = []
    rows = df.to_dict(orient='records')

    for dialogue, d_id, row in zip(df[dialogue_column], df['encounter_id'], rows):
        a_note = {'encounter_id': d_id}
        divisions = {}
        seeding_form = json.loads(dialogue)

        for each_title_detail in seeding_form:
            division = each_title_detail.get('division')
            title = each_title_detail.get('title')

            if not division or division not in ['subjective', 'objective_exam', 'objective_results',
                                                'assessment_and_plan']:
                log.warning(f"Invalid or missing division in seeding form: {division}")
                continue

            divisions.setdefault(division, {})
            divisions[division].setdefault(title, title.upper() + '\n')

            summarizer = each_title_detail.get('summarizer')
            if summarizer:
                method_name = summarizer.get('method')
                method = globals().get(method_name)
                if not method:
                    log.error(f"Method '{method_name}' not found!")
                    continue

                data = ''  # Initialize data variable
                if 'retrieval_result' in inspect.signature(method).parameters:
                    if 'retrieval_result' not in each_title_detail:
                        log.error(f"Expected 'retrieval_result' in seeding form but not found for {title}")
                        continue

                    retrieval_result = RetrievalResult(**each_title_detail['retrieval_result'])
                    other_kwargs = {k: v for k, v in summarizer.items() if k != 'method'}
                    data = method(retrieval_result=retrieval_result, **other_kwargs)

                elif 'row' in inspect.signature(method).parameters:
                    other_kwargs = {k: v for k, v in summarizer.items() if k != 'method'}
                    data = method(row=row, **other_kwargs)

                if isinstance(data, str) and data.strip():
                    print(f"Data for title '{title}': {data}")  # Debug line
                    divisions[division][title] += '\n' + data

        for division_name in ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']:
            a_note[division_name] = "\n\n".join(
                divisions.get(division_name, {}).values()) if division_name in divisions else '---NONE---'

        a_note['note'] = "\n\n".join(
            a_note[div] for div in ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan'])
        notes.append(a_note)

    return {'output': pd.DataFrame(notes)}


if __name__ == "__main__":
    # Sample dialogue data in JSON format
    dialogue_json = json.dumps([
        {
            "division": "subjective",
            "title": "Headache and Dizziness",
            "summarizer": {
                "prompt": "diagnosis: ",
                "max_length": 30,
                "min_length": 1,
                "model": "amagzari/bart-large-xsum-finetuned-samsum-v2",
                "method": "summarize",
                "device": "cpu"
            },
            "retrieval_result": {
                "texts": [
                    "The patient reports a persistent headache and dizziness. They have been experiencing these symptoms for the past week. No significant changes in their medication regimen. They are worried about potential causes and are seeking advice on how to manage their symptoms."
                ],
                "scores": [1.0],
                "detailed_texts": ["Detailed examination shows no abnormal findings."],
                "detailed_scores": [0.9]
            }
        }
    ])

    # Sample DataFrame
    data = {
        'dialogue': [dialogue_json],
        'index_column': [1],
    }

    df = pd.DataFrame(data)

    result = note_complete(df, dialogue_column='dialogue', index_column='index_column')

    output_df = result['output']
    output_df.to_csv('notes_output.csv', index=False)

    print("Results have been saved to 'notes_output.csv'")
