import pandas as pd


def chunk_dialogue(dialogue, max_tokens):
    # Split the dialogue into individual utterances using newlines
    utterances = dialogue.split("\n")

    chunks = []
    current_chunk = []
    current_token_count = 0

    # Process each utterance
    for utterance in utterances:
        # Tokenize the utterance by splitting on whitespace
        utterance_tokens = utterance.split()
        utterance_token_count = len(utterance_tokens)

        # If adding this utterance exceeds the max token limit, finalize the current chunk
        if current_token_count + utterance_token_count > max_tokens:
            chunks.append("\n".join(current_chunk))  # Add the current chunk to the chunks list
            current_chunk = [utterance]  # Start a new chunk
            current_token_count = utterance_token_count
        else:
            # Add the utterance to the current chunk
            current_chunk.append(utterance)
            current_token_count += utterance_token_count

    # Add the last chunk if there is any
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def chunk_dialogues_from_df(df_group, dialogue_column, max_tokens):
    # Initialize a list to store the chunked dialogues
    chunked_data = []

    # Process each dialogue in the DataFrame
    for index, row in df_group.iterrows():
        dialogue = row[dialogue_column]
        dataset = row['dataset']  # Extract dataset metadata
        encounter_id = row['encounter_id']  # Extract encounter_id metadata
        note = row['note']  # Extract note metadata
        # chunk handling
        dialogue_chunks = chunk_dialogue(dialogue, max_tokens)

        # Add each chunk to the chunked_data list along with the dialogue index
        for chunk_num, chunk in enumerate(dialogue_chunks):
            chunked_data.append({
                'dataset': dataset,
                'encounter_id': encounter_id,
                'note': note,
                'dialogue_id': index,
                'chunk_num': chunk_num + 1,
                'chunk_text': chunk
            })

    # Create a new DataFrame from the chunked data
    chunked_df = pd.DataFrame(chunked_data)

    return chunked_df