import re
from typing import *
import pandas as pd

FULL_CONTRACTIONS = {
    "\\bhigh blood pressure\\b": "hypertension",
    "\\bhep c\\b": "hepatitis c",
    "\\bchf\\b": "congestive heart failure",
    "\\betoside\\b": "etoposide",
    "\\regulin\\b": "reglan",
    "\\bcisplestan\\b": "cisplatin",
    "\\bfollowup\\b": "follow up",
    "\\ba day\\b": "daily",
    "\\bmilligram(s)?\\b": "mg",
    "\\bmilliram(s)?\\b": "mg",
    "\\bkilogram(s)?\\b": "kg",
    "\\bgram(s)?\\b": "g",
    "\\bpound(s)?\\b": "lb",
    "\\bounce(s)?\\b": "oz",
    "\\spercent\\b": "%",
    "\\btwenty-one\\b": "21",
    "\\btwenty-two\\b": "22",
    "\\btwenty-three\\b": "23",
    "\\btwenty-four\\b": "24",
    "\\btwenty-five\\b": "25",
    "\\btwenty-six\\b": "26",
    "\\btwenty-seven\\b": "27",
    "\\btwenty-eight\\b": "28",
    "\\btwenty-nine\\b": "29",
    "\\bthirty-one\\b": "31",
    "\\bthirty-two\\b": "32",
    "\\bthirty-three\\b": "33",
    "\\bthirty-four\\b": "34",
    "\\bthirty-five\\b": "35",
    "\\bthirty-six\\b": "36",
    "\\bthirty-seven\\b": "37",
    "\\bthirty-eight\\b": "38",
    "\\bthirty-nine\\b": "39",
    "\\bforty-one\\b": "41",
    "\\bforty-two\\b": "42",
    "\\bforty-three\\b": "43",
    "\\bforty-four\\b": "44",
    "\\bforty-five\\b": "45",
    "\\bforty-six\\b": "46",
    "\\bforty-seven\\b": "47",
    "\\bforty-eight\\b": "48",
    "\\bforty-nine\\b": "49",
    "\\bfifty-one\\b": "51",
    "\\bfifty-two\\b": "52",
    "\\bfifty-three\\b": "53",
    "\\bfifty-four\\b": "54",
    "\\bfifty-five\\b": "55",
    "\\bfifty-six\\b": "56",
    "\\bfifty-seven\\b": "57",
    "\\bfifty-eight\\b": "58",
    "\\bfifty-nine\\b": "59",
    "\\bsixty-one\\b": "61",
    "\\bsixty-two\\b": "62",
    "\\bsixty-three\\b": "63",
    "\\bsixty-four\\b": "64",
    "\\bsixty-five\\b": "65",
    "\\bsixty-six\\b": "66",
    "\\bsixty-seven\\b": "67",
    "\\bsixty-eight\\b": "68",
    "\\bsixty-nine\\b": "69",
    "\\bseventy-one\\b": "71",
    "\\bseventy-two\\b": "72",
    "\\bseventy-three\\b": "73",
    "\\bseventy-four\\b": "74",
    "\\bseventy-five\\b": "75",
    "\\bseventy-six\\b": "76",
    "\\bseventy-seven\\b": "77",
    "\\bseventy-eight\\b": "78",
    "\\bseventy-nine\\b": "79",
    "\\beighty-one\\b": "81",
    "\\beighty-two\\b": "82",
    "\\beighty-three\\b": "83",
    "\\beighty-four\\b": "84",
    "\\beighty-five\\b": "85",
    "\\beighty-six\\b": "86",
    "\\beighty-seven\\b": "87",
    "\\beighty-eight\\b": "88",
    "\\beighty-nine\\b": "89",
    "\\bninety-one\\b": "91",
    "\\bninety-two\\b": "92",
    "\\bninety-three\\b": "93",
    "\\bninety-four\\b": "94",
    "\\bninety-five\\b": "95",
    "\\bninety-six\\b": "96",
    "\\bninety-seven\\b": "97",
    "\\bninety-eight\\b": "98",
    "\\bninety-nine\\b": "99",
    "\\beleven\\b": "11",
    "\\btwelve\\b": "12",
    "\\bthirteen\\b": "13",
    "\\bfourteen\\b": "14",
    "\\bfifteen\\b": "15",
    "\\bsixteen\\b": "16",
    "\\bseventeen\\b": "17",
    "\\beighteen\\b": "18",
    "\\bnineteen\\b": "19",
    "\\btwenty one\\b": "21",
    "\\btwenty two\\b": "22",
    "\\btwenty three\\b": "23",
    "\\btwenty four\\b": "24",
    "\\btwenty five\\b": "25",
    "\\btwenty six\\b": "26",
    "\\btwenty seven\\b": "27",
    "\\btwenty eight\\b": "28",
    "\\btwenty nine\\b": "29",
    "\\bthirty one\\b": "31",
    "\\bthirty two\\b": "32",
    "\\bthirty three\\b": "33",
    "\\bthirty four\\b": "34",
    "\\bthirty five\\b": "35",
    "\\bthirty six\\b": "36",
    "\\bthirty seven\\b": "37",
    "\\bthirty eight\\b": "38",
    "\\bthirty nine\\b": "39",
    "\\bforty one\\b": "41",
    "\\bforty two\\b": "42",
    "\\bforty three\\b": "43",
    "\\bforty four\\b": "44",
    "\\bforty five\\b": "45",
    "\\bforty six\\b": "46",
    "\\bforty seven\\b": "47",
    "\\bforty eight\\b": "48",
    "\\bforty nine\\b": "49",
    "\\bfifty one\\b": "51",
    "\\bfifty two\\b": "52",
    "\\bfifty three\\b": "53",
    "\\bfifty four\\b": "54",
    "\\bfifty five\\b": "55",
    "\\bfifty six\\b": "56",
    "\\bfifty seven\\b": "57",
    "\\bfifty eight\\b": "58",
    "\\bfifty nine\\b": "59",
    "\\bsixty one\\b": "61",
    "\\bsixty two\\b": "62",
    "\\bsixty three\\b": "63",
    "\\bsixty four\\b": "64",
    "\\bsixty five\\b": "65",
    "\\bsixty six\\b": "66",
    "\\bsixty seven\\b": "67",
    "\\bsixty eight\\b": "68",
    "\\bsixty nine\\b": "69",
    "\\bseventy one\\b": "71",
    "\\bseventy two\\b": "72",
    "\\bseventy three\\b": "73",
    "\\bseventy four\\b": "74",
    "\\bseventy five\\b": "75",
    "\\bseventy six\\b": "76",
    "\\bseventy seven\\b": "77",
    "\\bseventy eight\\b": "78",
    "\\bseventy nine\\b": "79",
    "\\beighty one\\b": "81",
    "\\beighty two\\b": "82",
    "\\beighty three\\b": "83",
    "\\beighty four\\b": "84",
    "\\beighty five\\b": "85",
    "\\beighty six\\b": "86",
    "\\beighty seven\\b": "87",
    "\\beighty eight\\b": "88",
    "\\beighty nine\\b": "89",
    "\\bninety one\\b": "91",
    "\\bninety two\\b": "92",
    "\\bninety three\\b": "93",
    "\\bninety four\\b": "94",
    "\\bninety five\\b": "95",
    "\\bninety six\\b": "96",
    "\\bninety seven\\b": "97",
    "\\bninety eight\\b": "98",
    "\\bninety nine\\b": "99",
    "\\bone hundred\\b": "100",
    "\\ba hundred\\b": "100",
    "\\btwo hundred\\b": "200",
    "\\bthree hundred\\b": "300",
    "\\bfour hundred\\b": "400",
    "\\bfive hundred\\b": "500",
    "\\bsix hundred\\b": "600",
    "\\bseven hundred\\b": "700",
    "\\beight hundred\\b": "800",
    "\\bnine hundred\\b": "900",
    "\\ba thousand\\b": "1000",
    "\\beighties\\b": "80s",
    "\\bone\\b": "1",
    "\\btwo\\b": "2",
    "\\bthree\\b": "3",
    "\\bfour\\b": "4",
    "\\bfive\\b": "5",
    "\\bsix\\b": "6",
    "\\bseven\\b": "7",
    "\\beight\\b": "8",
    "\\bnine\\b": "9",
    "\\bten\\b": "10",
    "\\btwenty\\b": "20",
    "\\bthirty\\b": "30",
    "\\bforty\\b": "40",
    "\\bfifty\\b": "50",
    "\\bsixty\\b": "60",
    "\\bseventy\\b": "70",
    "\\beighty\\b": "80",
    "\\bninety\\b": "90",
    "(\\d+)\\s+over\\s+(\\d+)": "\\1/\\2",
    "(\\d+)\\s+\\.\\s+(\\d+)": "\\1.\\2",
    "do you have any question[^\\n?\\.]*[?\\.\\n]": "",
    "([?\\.\\n])[^\\n?\\.]*(questions|more question|thanks? )[^\\n?\\.]*[?\\.\\n]": "\\1",
    "(?!.*(continue|increase|recommend|keep|ekg|blood|prescribe))((so( what)?|and) )?i am going to(( \\w+){0,3}( and) \\w+|( \\w+){0,3})( (to|another|a(n)?))?": "",
    "((so( what)?|and) )?i am going to( ((just )?go ahead( and)?|call me( and then)?|do is (\\w+)|have you|order|))*": "",
    "(?!.*(continue|increase|prescribe|blood|stick|dehydration|watch))((so( what)?|and) )?i( (do|really|not|also|just|would))* (want(ed)? you|want(ed)?)( to)?(( \\w+){0,3}( and) \\w+|( \\w+){0,3})( (to|another|a))?": "",
    "((so( what)?|and) )?i( (do|really|not|also|just|would))* (want(ed)? you|want(ed)?)( to)?( ((just )?go ahead( and)?|call me( and then)?|do is (\\w+)|have you|order|))*": "",
    "(?!\\n?.*(procedure|follow[ -]up|dose|prescribe))([?\\.\\n])[^\\n?\\.]*i\\'ll[^\\n?\\.]*[?\\.\\n]": "\\2",
    "prescribe you": "prescribed the patient",
    "(((as far )?as|so|what|which|do) )?you know( (what|is))?": "",
    "do you have any question[^\\n\\?\\.]*[\\?\\.\\n]": "",
    "([\\?\\.\\n])[^\\n\\?\\.]*(questions|more question|thanks? )[^\\n\\?\\.]*[\\?\\.\\n]": "\\1",
    "(?!\\n?.*(procedure|follow[ -]up|dose|prescribe))([\\?\\.\\n])[^\\n\\?\\.]*i\\'ll[^\\n\\?\\.]*[\\?\\.\\n]": "\\2",
    "how are you (feeling right now\\?|doing,)": "",
    ", you know ?": ",",
    ", you know ,": ",",
    ", you know,": ",",
    "you know ,": "",
    "you know,": "",
    "\bi have\b": "the patient has",
    "\bi am\b": "the patient is",
    "\bi would\b": "the patient would",
    "\bi\b": "the patient",
    "\bme\b": "the patient",
    "\bmy\b": "his / her",
    "you have": "the patient has",
    "you( are|\\'re)": "the patient is",
    "you do": "the patient does",
    "(you|you guys)\\b": "the patient",
    "your": "the patient's",
    "(?![^?]*\\b(blood|recommended|taking|surgical|flu|fever|vaccine|keeping up|smok(ing|ed|e)|hurt|pain|motion|steroids?|ibuprofen|gout)\\b)([\\n\\?\\.])[^\\n\\?\\.]*\\?": "\\2",
    "(?<!(...as|going|doing) )((\\bum|(\\bum )?well|(so )?okay|great|yeah|(so )?all right(-y)?|(so )?alright|\\bso( and)?|\\buh|(so )?first of all), ?)+": "",
    ", okay\\.": ".",
    "hey,? dragon,?( finalize the note\\.)?": "",
    "\\n{3,}": "\\n\\n",
    "(\\s) +": "\\1",
    " *([\\.,])( ?[\\.,])+": "\\1"
}


def replace_terms(text: str, FULL_CONTRACTIONS: Dict) -> str:
    """
    Replace occurrences of terms in the given text using a dictionary of replacements.
    """
    for pattern, replacement in FULL_CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def preprocess_data(df: pd.DataFrame,
                    dialogue_col: str,
                    FULL_CONTRACTIONS: Dict) -> dict:
    """
    Processes dialogue data from a DataFrame by replacing terms using a provided dictionary of contractions.

    """

    def process_row(row):
        text = row[dialogue_col]
        divisions = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]

        for division in divisions:
            if division in row:
                row[division] = replace_terms(row[division], FULL_CONTRACTIONS)

        row['note'] = replace_terms(text, FULL_CONTRACTIONS)
        return row

    df = df.apply(process_row, axis=1)

    return df


if __name__ == "__main__":
    # Sample data for testing
    data = {
        'encounter_id': [1, 2, 3],
        'dialogue': [
            'You have high blood pressure and takes etoposide daily.',
            'Follow up with the patient about their hepatitis c treatment.',
            'The patient reports pain in their hands and feet.'
        ],
        'subjective': [
            'High blood pressure and etoposide daily.',
            'Hepatitis c treatment follow-up.',
            'Pain in hands and feet.'
        ],
        'objective_exam': [
            'No abnormalities observed.',
            'Routine check-up.',
            'No significant findings.'
        ],
        'objective_results': [
            'Blood pressure high.',
            'Hepatitis c levels stable.',
            'No new symptoms.'
        ],
        'assessment_and_plan': [
            'Monitor blood pressure.',
            'Continue hepatitis c treatment.',
            'Assess pain management.'
        ]
    }

    df = pd.DataFrame(data)

    # Preprocess data
    processed_df = preprocess_data(df, 'dialogue', FULL_CONTRACTIONS)

    # Save original and processed DataFrames to CSV files
    df.to_csv('original_data.csv', index=False)
    processed_df.to_csv('processed_data.csv', index=False)

    print("Original and processed data have been saved to 'original_data.csv' and 'processed_data.csv'.")
