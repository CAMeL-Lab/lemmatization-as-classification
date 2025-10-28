# === Mapping LPG Encodings to Lex, POS, and StemGloss ===
def map_lpg_info(row, final_s31_lex):
    """
    Maps a list of LPG encoded IDs to their corresponding Lex, POS, and stemgloss values
    using the final S31 lexicon.

    Args:
        row (list): A list of LPG-encoded IDs (strings or integers).
        final_s31_lex (pd.DataFrame): DataFrame containing 'LPG_encoded', 'Lex', 'POS', 'stemgloss'.

    Returns:
        list[dict]: A list of dictionaries mapping LPG ID to its linguistic features.
    """
    mapped_info = []

    for item in row:
        lpg_id = int(item)
        match = final_s31_lex[final_s31_lex['LPG_encoded'] == lpg_id][['Lex', 'POS', 'stemgloss']]
        if not match.empty:
            lex, pos, stemgloss = match.iloc[0]
            mapped_info.append({
                'LPG': lpg_id,
                'Lex': lex,
                'POS': pos,
                'stemgloss': stemgloss
            })

    return mapped_info

# === Chunking Sentences Longer Than Tokenizer Limit ===
def chunk_sentence(sentence, tokenizer, max_length=512):
    """
    Breaks a sentence into chunks such that tokenized input does not exceed max_length.

    Args:
        sentence (list): List of words (tokens) in a sentence.
        tokenizer: HuggingFace tokenizer instance.
        max_length (int): Maximum number of tokens allowed by model.

    Returns:
        list[list]: List of sentence chunks (each chunk is a list of words).
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for word in sentence:
        token_len = len(tokenizer(word, return_tensors="pt", truncation=True, padding=True, max_length=512)["input_ids"][0])

        if current_length + token_len <= max_length - 2:
            current_chunk.append(word)
            current_length += token_len
        else:
            chunks.append(current_chunk)
            current_chunk = [word]
            current_length = token_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
