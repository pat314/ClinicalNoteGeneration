"""
This is a courtesy script that outpus additional columns for different divisions of the note.
Please adjust according to your own outputs. Make sure you use our supplied EMPTY_TAG if you 
deem that a certain encouner should not have a particula division.
"""
import sys
import os
import re

import pandas as pd

from .sectiontagger import SectionTagger
section_tagger = SectionTagger()

SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']
EMPTY_TAG = '#####EMPTY#####'

def add_section_divisions( row, note_column ):
    
    text = row[ note_column ]
    detected_divisions = section_tagger.divide_note_by_metasections( text )
    for detected_division in detected_divisions:
        label, _, _, start, _, end = detected_division
        row[ label ] = text[start:end]

    return row


if __name__ == "__main__" :
    """
    Usage: python format_seperate_divisions.py <input-csv> <note-column> <output-csv>  
    """
    if len( sys.argv ) > 1 :
        fn = sys.argv[1]
        note_column = [ sys.argv[2] if len(sys.argv)>2 else 'note' ][0]
        fn_out = [ sys.argv[3] if len(sys.argv)>3 else fn.replace( '.csv', '_div.csv' ) ][0]
    else :
        print( 'usage: python format_seperate_divisions.py <input-csv> <note-column> <output-csv>')
        sys.exit( 0 )

    df = pd.read_csv( fn )

    df = df.apply( lambda row: add_section_divisions( row, note_column ), axis=1 )
    
    for division in SECTION_DIVISIONS :
        if division in df.columns :
            continue
        else :
            df[ division ] = EMPTY_TAG
    
    df.fillna( EMPTY_TAG, inplace=True)

    print( 'outputting to: %s' %fn_out )
    df.to_csv( fn_out, index=False )
