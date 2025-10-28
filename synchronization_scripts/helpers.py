import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
import re
import Levenshtein
from camel_tools.utils.charmap import CharMapper


ud_mada_pos_mapping = {
    'NOUN': 'noun',
    'NOUN+PART': 'noun',
    'CCONJ+NOUN': 'noun',
    'NOUN+PRON': 'noun',
    'PROPN': 'noun_prop',
    'VERB': 'verb',
    'ADP+NOUN': 'noun',
    'PART+NOUN+PRON': 'noun',
    'PART+NOUN': 'noun',
    'ADJ': 'adj',
    'CCONJ+PROPN': 'noun_prop',
    'CCONJ+ADJ' : 'adj',
    'CCONJ+ADP+NOUN' : "noun",
    'ADP+PROPN' : "noun_prop",
    'AUX+VERB' : 'verb',
    'VERB+PRON' : 'verb',
    'PART+ADJ' : 'adj',
}

static_stuff_to_make_sure = [
                                ('وَإِيَّىَ', 'إِيّا', 'part', 'to;for;(accus.)'),
                                ('ءَأَنتُمْ', 'أَنْتُم', 'pron' , 'you_[masc.pl.]'),
                                ('لَهُوَ', 'هُوَ', 'pron', 'it/he'),
                                ('هَأَنتُمْ', 'أَنْتُم', 'pron', 'you_[masc.pl.]'),
                                ('ءَأَنتَ', 'أَنْتَ', 'pron', 'you_[masc.sg.]'),
                                ('أَفَأَنتَ', 'أَنْتَ', 'pron', 'you_[masc.sg.]'),
                                ('لَأَنتَ', 'أَنْتَ', 'pron', 'you_[masc.sg.]'),
                                ('لَنَحْنُ', 'نَحْنُ', 'pron', 'we'),
                                ('فَإِيَّىَ', 'إِيّا', 'part', 'to;for;(accus.)'),
                                ('أَفَهُمْ', 'هُم', 'pron', 'they_[masc.pl]'),
                                ('أَفَهُمُ', 'هُم', 'pron', 'they_[masc.pl]'),
                                ('أَفَأَنتُمْ', 'أَنْتُم', 'pron', 'you_[masc.pl.]'),
                                ('لَهِىَ', 'هِيَ', 'pron', 'it/they/she'),
                                ('أَنَحْنُ', 'نَحْنُ', 'pron', 'we'),
                                ('أَهُمْ', 'هُم', 'pron', 'they_[masc.pl]'),
                                ('لَأَنتُمْ', 'أَنْتُم', 'pron', 'you_[masc.pl.]'),
                                ('أَهُمْ', 'هُم', 'pron', 'they_[masc.pl]'),
                            ]

static_stuff_to_with_ours = [('adv',	'هٰكَذا',	'ذَا'),
('adv',	'رُبَّما',	'رُبَّ'),
('adv_rel',	'كُلَّما',	'كُلّ'),
('conj_sub',	'لَوْلا',	'لَوْ'),
('part_det',	'ال',	'وَآَّل'),
('part_focus',	'أَمّا',	'إِنْ'),
('prep',	'بِ',	'بِي'),
('prep',	'لِ',	'لِي'),
('pron',	'هِيَ',	'هُوَ'),
('pron',	'هُم',	'هُوَ'),
('pron',	'هُما',	'هُوَ'),
('pron_dem',	'ذٰلِكَ',	'ذَا'),
('pron_dem',	'هٰذا',	'ذَا'),
('pron_dem',	'ذا',	'ذُو')]

# Convert the static list into a dictionary for faster lookup
static_dict = {item[0]: item[1:] for item in static_stuff_to_make_sure}

back_off = 'NONE'
db = MorphologyDB('/Users/mms10094/Documents/Coding Stuff/Camel lemma analysis/Morphology DB/calima-msa-s31_0.4.2.utf8.db')
# db = MorphologyDB('/Users/mms10094/Documents/Coding Stuff/Camel lemma analysis/Morphology DB/camel_morph_msa_v1.1.0.db')
calima_analyzer = Analyzer(db, back_off)
bert_disambig = BERTUnfactoredDisambiguator.pretrained('msa', top = 5000, pretrained_cache=False)
bert_disambig._analyzer = calima_analyzer

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw_mapper = CharMapper.builtin_mapper('ar2bw')

def lex_transliterate(text):
    return bw2ar.map_string(text) if text != "UNK" else text

def disambig_output(bert_disambig, calima_analyzer, df):
    """
    This function runs disambiguation using an unfactored BERT disambiguation model (`bert_disambig`) 
    on each sentence in the given DataFrame (`df`). It processes the text and selects specific features 
    from the model's output.

    - The function iterates over unique sentence indices in `df`, tokenizing each sentence.
    - It runs disambiguation on the tokenized sentence and extracts relevant features.
    - If a word contains a dagger alef (ٰ), the function runs disambiguation twice:
      1. With the dagger alef as it is.
      2. With the dagger alef normalized to a regular alef (ا).
    - The results are collected and stored in a DataFrame with selected features.

    Parameters:
        bert_disambig: The unfactored BERT disambiguation model.
        df (pd.DataFrame): A DataFrame containing tokenized sentences with a 'sentence_index' & 'word_index' column.

    Returns:
        pd.DataFrame: A DataFrame containing the disambiguation results with selected features.
    """
        
    selected_features = ['lex', 'pos', 'prc3', 'prc2', 'prc1', 'prc0', 'per', 'asp', 'vox', 'mod', 'form_gen', 'form_num', 'stt', 'cas', 'enc0', 'stemgloss', 'gloss', 'source', 'lex_logprob', 'pos_lex_logprob', 'ud']
    
    rows = []
    dagger_alef = "ٰ"
    normal_alef = "ا"

    for i in tqdm(range(0, len(df['sentence_index'].unique())), desc="Processing sentences"):
        sentence_index = df['sentence_index'].unique()[i]
        sentence_df = df[df['sentence_index'] == sentence_index]
        tokenized_sent = list(sentence_df['word'])
        original_word_indices = list(sentence_df.index)

        # Identify words with dagger alef
        words_with_dagger_alef = [(idx, word) for idx, word in enumerate(tokenized_sent) if dagger_alef in word]

        def process_sentence(sentence_tokens, original_indices, sentence_idx):
            """Process a sentence using BERT disambiguation and store results."""
            # testing = bert_disambig.disambiguate(sentence_tokens)
            testing = calima_analyzer.analyze_words(sentence_tokens)
            local_rows = []
            
            for local_word_index, word_index in enumerate(tqdm(testing, desc=f"Processing words in sentence {sentence_idx}", leave=False)):
                # max_score = word_index.analyses[0].score if word_index.analyses else 0

                if len(word_index.analyses) == 0:
                    # If no analyses, fill analysis details with NaNs
                    analysis_details = {key: np.nan for key in selected_features}
                    original_index = original_indices[local_word_index]
                    row = [sentence_idx, local_word_index, original_index, word_index.word] + list(analysis_details.values())
                    local_rows.append(row)
                else:
                    # for analysis_index in word_index.analyses:
                    for analysis_index in word_index.analyses:
                        word = word_index.word
                        # score = analysis_index.score

                        # Extract analysis details with exact column name matching
                        # analysis_details = {key: analysis_index.analyses[key] if key in analysis_index.analyses else None for key in selected_features}
                        analysis_details = {key: analysis_index[key] if key in analysis_index else None for key in selected_features}
                        
                        # Get the original index
                        original_index = original_indices[local_word_index]
                        
                        # Store result
                        # row = [sentence_idx, local_word_index, original_index, word, score, max_score] + list(analysis_details.values())
                        row = [sentence_idx, local_word_index, original_index, word] + list(analysis_details.values())
                        local_rows.append(row)
            
            return local_rows

        # First run (normal processing)
        rows.extend(process_sentence(tokenized_sent, original_word_indices, sentence_index))

        # Process only words that contain dagger alef
        if words_with_dagger_alef:
            modified_tokens = tokenized_sent.copy()
            modified_word_indices = []

            for word_idx, word in words_with_dagger_alef:
                modified_tokens[word_idx] = word.replace(dagger_alef, normal_alef)
                modified_word_indices.append(original_word_indices[word_idx])  # Keep track of indices of modified words
            
            # Process only the modified words
            modified_results = process_sentence(modified_tokens, original_word_indices, sentence_index)
            
            # Filter only modified words in the results and append them to the output
            for row in modified_results:
                if row[3] in [w.replace(dagger_alef, normal_alef) for _, w in words_with_dagger_alef]:  # Ensure only modified words are added
                    rows.append(row)

    # Define the output DataFrame
    # columns = ['sentence_index', 'word_index', 'original_index', 'word', 'score', 'max_score'] + selected_features
    columns = ['sentence_index', 'word_index', 'original_index', 'word'] + selected_features
    output_df = pd.DataFrame(rows, columns=columns)
    
    return output_df

def clean_words_with_camel_arclean(df, words_column, words_file="words.txt", output_file=f"cleaned_words.txt"):
    """
    Cleans a specified column in a DataFrame using the camel_arclean CLI tool.
    
    Steps:
    1. Saves the column to a text file.
    2. Runs the camel_arclean CLI command on the text file.
    3. Reads the cleaned output and updates the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the words to clean.
        words_column (str): The name of the column to clean.
        words_file (str, optional): Name of the temporary file for storing words (default: "words.txt").
        output_file (str, optional): Name of the output file from camel_arclean (default: "zabuc_cleaned_words.txt").
    
    Returns:
        pd.DataFrame: A new DataFrame with the cleaned words in the specified column.
    """
    
    # Step 1: Save the words column to a file
    df[words_column].to_csv(words_file, index=False, header=False)

    # Step 2: Run camel_arclean CLI on the text file
    command = f"camel_arclean -o {output_file} {words_file}"
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("CLI command ran successfully.")
        print(result.stdout)  # Optional: Print standard output from the command
    except subprocess.CalledProcessError as e:
        print("Error running the CLI command.")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise  # Re-raise the exception to stop the script

    # Step 3: Read the cleaned output and update the DataFrame
    try:
        with open(output_file, "r") as f:
            cleaned_words = f.read().splitlines()

        df[words_column] = cleaned_words
        print("Cleaned words successfully updated in the DataFrame.")
        
        return df  # Return the updated DataFrame

    except FileNotFoundError:
        print(f"Error: Output file {output_file} not found. The CLI command may have failed.")
        raise

def normalize_text(text, data):
    """
    This function normalizes and applies adjustments to input words by removing or modifying specific diacritic patterns.
    The function also tracks the number of times each transformation is applied.

    Parameters:
        text (str): The input word to be normalized.
        data (str): Context of the text (e.g., 'quranic') to apply specific transformations.

    Returns:
        tuple: (normalized_text, counts of applied transformations)
        
    """

    if pd.isna(text):
        return text, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Added two new counters

    transformation_counts = {
        "remove_fatha_before_alef": 0,
        "remove_damma_before_waw": 0,  
        "remove_kasra_before_yaa": 0,  
        "remove_fatha_before_superscript_alef": 0,
        "replace_superscript_alef_with_fatha": 0,
        "move_tanween_to_last_letter": 0,
        "remove_diacritics_last_letter": 0,
        "replace_alef_wasla_kasra": 0,
        "fix_shadda_order": 0,  
        "fix_sun_letter_shadda": 0  
    }

    #replace kasra alef_layena with kasra yaaa2
    text =  re.sub(r'ِى', 'ِي', text)
    
    if data == 'quranic':
        text = re.sub(r'\^', '', text)
    
    # Check if a diacritic appears before Shadda and fix order
    if re.search(r'[\u064E\u064F\u0650\u0652]\u0651', text):  
        text = re.sub(r'([\u064E\u064F\u0650\u0652])(\u0651)', r'\2\1', text)  
        transformation_counts["fix_shadda_order"] += 1

    # Replace Alef Wasla with Kasra (ٱِ) with Alef with Kasra (اِ)
    if re.search(r'ٱِ', text):
        text = re.sub(r'ٱِ', 'اِ', text)
        transformation_counts["replace_alef_wasla_kasra"] += 1

    # Remove Fatha before Alef only if Alef has no diacritics
    if re.search(r'َ(ا)(?![\u064B-\u0652])', text):
        text = re.sub(r'َ(ا)(?![\u064B-\u0652])', r'\1', text)
        transformation_counts["remove_fatha_before_alef"] += 1

    # Remove Damma before Waw only if Waw has no diacritics
    if re.search(r'ُ(و)(?![\u064B-\u0652])', text):
        text = re.sub(r'ُ(و)(?![\u064B-\u0652])', r'\1', text)
        transformation_counts["remove_damma_before_waw"] += 1

    # Remove Kasra before Yaa only if Yaa has no diacritics
    if re.search(r'ِ([يى])(?![\u064B-\u0652])', text):
        text = re.sub(r'ِ([يى])(?![\u064B-\u0652])', r'\1', text)
        transformation_counts["remove_kasra_before_yaa"] += 1

    # Remove Fatha before superscript Alef
    if re.search(r'َ(?=ٰ)', text):
        text = re.sub(r'َ(?=ٰ)', '', text)
        transformation_counts["remove_fatha_before_superscript_alef"] += 1

    # Replace superscript Alef with Fatha
    if re.search(r'ٰ', text):
        text = re.sub(r'ٰ', 'َ', text)
        transformation_counts["replace_superscript_alef_with_fatha"] += 1

    # Move the tanween from before last to the last letter
    match = re.search(r'(\w)([ًٌٍ])(\w[\u064B-\u0652]*)$', text)
    if match:
        second_last_letter = match.group(1)
        tanween = match.group(2)
        last_letter_with_diacritics = match.group(3)
        text = text[:match.start()] + second_last_letter + last_letter_with_diacritics + tanween
        transformation_counts["move_tanween_to_last_letter"] += 1

    # Remove all diacritics on the last letter except shadda
    if re.search(r'([^\sًٌٍَُِّْ])([ًٌٍَُِْ]*ّ?[ًٌٍَُِْ]*)$', text):
        text = re.sub(r'([^\sًٌٍَُِّْ])([ًٌٍَُِْ]*)(ّ[ًٌٍَُِْ]*|)([ًٌٍَُِْ]*)$', 
        lambda m: m.group(1) + m.group(3), text)
        transformation_counts["remove_diacritics_last_letter"] += 1

    # Fix Sun Letter + Shadda issue
    sun_letter_shadda_pattern = re.compile(r'^([تثدذرزسشصضطظلن])\u0651([\u064E\u064F\u0650\u0652\u064B\u064C\u064D]?)')

    match1 = sun_letter_shadda_pattern.search(text)
    if match1:
        sun_letter = match1.group(1)
        diacritic = match1.group(2)  # Get the diacritic if it exists

        # If no diacritic is present, keep the Shadda as it is
        if not diacritic:
            replacement = sun_letter + '\u0651'  # Keep the Shadda
        else:
            # Handle diacritic and Tanween variations
            if diacritic in ['\u064E', '\u064B']:  # Fatha / Fathatan
                replacement = sun_letter + diacritic
            elif diacritic in ['\u064F', '\u064C']:  # Damma / Dammatan
                replacement = sun_letter + diacritic
            elif diacritic in ['\u0650', '\u064D']:  # Kasra / Kasratan
                replacement = sun_letter + diacritic
            elif diacritic == '\u0652':  # Sukun
                replacement = sun_letter + '\u0652'
            transformation_counts["fix_sun_letter_shadda"] += 1

        # Perform replacement in text only at the first occurrence
        text = sun_letter_shadda_pattern.sub(replacement, text)
        
        
    text =  re.sub(r'ٱ', 'ا', text)
    
    return (text, 
            transformation_counts["remove_fatha_before_alef"],
            transformation_counts["remove_damma_before_waw"],
            transformation_counts["remove_kasra_before_yaa"],
            transformation_counts["remove_fatha_before_superscript_alef"],
            transformation_counts["replace_superscript_alef_with_fatha"],
            transformation_counts["move_tanween_to_last_letter"],
            transformation_counts["remove_diacritics_last_letter"],
            transformation_counts["replace_alef_wasla_kasra"],
            transformation_counts["fix_shadda_order"],
            transformation_counts["fix_sun_letter_shadda"])  

def fix_sentence_and_word_index(df):
    """
    This function ensures that:
    1. `sentence_index` values are sequential (starting from 0) even if there are missing numbers.
    2. `word_index` values within each sentence are sequential (starting from 0) if there are gaps.

    Steps:
    - Create a mapping to reassign `sentence_index` values sequentially.
    - Reassign `word_index` values within each sentence to ensure continuity.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing `sentence_index` and `word_index`.

    Returns:
        pd.DataFrame: A new DataFrame with fixed `sentence_index` and `word_index` values.
    """
    
    df = df.copy()
    
    # Ensure sentence indices are sequential (starting from 0)
    unique_sentence_indices = sorted(df['sentence_index'].unique())  # Get unique, sorted sentence indices
    sentence_index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_sentence_indices)}
    
    df['sentence_index'] = df['sentence_index'].map(sentence_index_mapping)  # Apply mapping
    
    # Fix word_index within each sentence
    new_word_indices = []
    
    for sentence_idx in df['sentence_index'].unique():
        mask = df['sentence_index'] == sentence_idx
        temp_df = df[mask].copy()
        
        # Generate a sequential word index starting from 0
        temp_df['word_index'] = range(len(temp_df))
        
        # Store updated values
        new_word_indices.extend(temp_df['word_index'].tolist())
    
    # Update the DataFrame with new word_index values
    df['word_index'] = new_word_indices
    
    return df

def remove_diacritics_and_symbols(text):
    """Remove diacritics and symbols from Arabic text."""
    return re.sub(r'[\u064B-\u065F\W]', '', text)  # Removes Arabic diacritics and non-word characters

def stemgloss_match_score_multi(stemgloss_output, stemgloss_df):
    if pd.isna(stemgloss_output) or pd.isna(stemgloss_df):
        return 0.0, 0.0, 0.0 

    # Remove all non-English characters and replace them with a space
    stemgloss_output = re.sub(r"[^a-zA-Z ]", " ", stemgloss_output)
    stemgloss_df = re.sub(r"[^a-zA-Z ]", " ", stemgloss_df)
    
    stemgloss_output = stemgloss_output.lower().strip()
    stemgloss_df = stemgloss_df.lower().strip()
    
    df_set = set(stemgloss_df.split())
    output_set = set(stemgloss_output.split())

    correct_matches = df_set.intersection(output_set)
    match_count = len(correct_matches)

    recall = match_count / len(df_set) if df_set else 0.0
    precision = match_count / len(output_set) if output_set else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return recall, precision, f1_score

def lex_match_score_multi(lex_output, lex_df):
    """Assign a score for lex based on exact match, diacritic removal, and normalized edit distance."""
    if pd.isna(lex_output) or pd.isna(lex_df):
        return 0.0 

    lex_output, lex_df = str(lex_output), str(lex_df)

    # Check for exact match
    if lex_output == lex_df:
        return 1.0

    # Remove diacritics and symbols for comparison
    lex_output_clean = remove_diacritics_and_symbols(lex_output)
    lex_df_clean = remove_diacritics_and_symbols(lex_df)

    # If cleaned versions do not match, return 0
    if lex_output_clean != lex_df_clean:
        return 0.0

    # Compute normalized edit distance for partial matches
    edit_distance = Levenshtein.distance(lex_output, lex_df)
    max_len = max(len(lex_output), len(lex_df))
    normalized_score = 1 - (edit_distance / max_len) if max_len > 0 else 0.0

    return normalized_score

def pos_match_score_multi(pos_output, pos_df):
    """Assign a score for pos based on equality."""
    return 1 if pos_output == pos_df else 0

def compute_highest_scoring_records_multi(df, salma_output_df, lex_pos_flag=False, lex_flag=False, lex_pos_stemgloss_flag=False, ref_tag='', input_tag='', data=''):
    # Merge both DataFrames on sentence_index and word_index
    merged_df = df.merge(salma_output_df, on=['sentence_index', 'word_index'], suffixes=(ref_tag, input_tag))
    
    if lex_flag == True:

        if data == 'nemlar':
            merged_df['lex_score'] = merged_df.apply(
            lambda row: lex_match_score_multi(row['adjusted_lex'+input_tag], row['adjusted_lex'+ref_tag]), axis=1
            )
        else:
            merged_df['lex_score'] = merged_df.apply(
                lambda row: lex_match_score_multi(
                    str(row['adjusted_lex' + input_tag]).strip() if pd.notna(row['adjusted_lex' + input_tag]) else '',
                    str(row['adjusted_lex' + ref_tag]).split()[0].strip()
                    if pd.notna(row['adjusted_lex' + ref_tag]) and len(str(row['adjusted_lex' + ref_tag]).split()) > 1
                    else str(row['adjusted_lex' + ref_tag]).strip() if pd.notna(row['adjusted_lex' + ref_tag]) else ''
                ), 
                axis=1
            )


        # Calculate sync score
        merged_df['sync_score'] = merged_df['lex_score']

        # Sort by sync_score and pos_lex_logprob to resolve ties
        merged_df.sort_values(by=['sentence_index', 'word_index', 'sync_score', 'pos_lex_logprob', 'lex_logprob'], ascending=[True, True, False, False, False], inplace=True)

        # # Get the highest scoring record for each sentence_index and word_index
        # highest_scores = merged_df.groupby(['sentence_index', 'word_index']).first().reset_index()

        # Select relevant columns
        # result = merged_df[
        #     ['sentence_index', 'word_index', 'word'+ref_tag, 'lex' + input_tag, 'pos', 'stemgloss',  'adjusted_lex' + input_tag, 'adjusted_lex'+ref_tag, 'lex'+ref_tag, 'lex_score', 'sync_score', 'score', 'max_score', 'pos_lex_logprob']
        # ]
        result = merged_df[
            ['sentence_index', 'word_index', 'word'+ref_tag, 'lex' + input_tag, 'pos', 'stemgloss',  'adjusted_lex' + input_tag, 'adjusted_lex'+ref_tag, 'lex'+ref_tag, 'lex_score', 'sync_score', 'pos_lex_logprob']
        ]
    
    if lex_pos_flag == True:

        merged_df['lex_score'] = merged_df.apply(
            lambda row: lex_match_score_multi(
                str(row['adjusted_lex' + input_tag]).strip() if pd.notna(row['adjusted_lex' + input_tag]) else '',
                str(row['adjusted_lex' + ref_tag]).split()[0].strip()
                if pd.notna(row['adjusted_lex' + ref_tag]) and len(str(row['adjusted_lex' + ref_tag]).split()) > 1
                else str(row['adjusted_lex' + ref_tag]).strip() if pd.notna(row['adjusted_lex' + ref_tag]) else ''
            ), 
            axis=1
        )
        

        
        merged_df['pos_score'] = merged_df.apply(
            lambda row: pos_match_score_multi(row['pos' + input_tag], row['pos' + ref_tag]), axis=1
        )

        # Calculate sync score
        merged_df['sync_score'] = merged_df['lex_score'] + merged_df['pos_score']

        # Sort by sync_score and pos_lex_logprob to resolve ties
        merged_df.sort_values(by=['sentence_index', 'word_index', 'sync_score', 'pos_lex_logprob'], ascending=[True, True, False, False], inplace=True)
        
        result = merged_df[
            ['sentence_index', 'word_index', 'word' + ref_tag, 'lex' + input_tag, 'pos' + input_tag, 'stemgloss',  'adjusted_lex' + input_tag, 'lex' + ref_tag, 'adjusted_lex' + ref_tag, 'pos' + ref_tag, 'lex_score', 'pos_score', 'sync_score', 'pos_lex_logprob', 'lex_logprob']
        ]
   
    if lex_pos_stemgloss_flag == True:

        merged_df['lex_score'] = merged_df.apply(
            lambda row: lex_match_score_multi(
                str(row['adjusted_lex' + input_tag]).strip() if pd.notna(row['adjusted_lex' + input_tag]) else '',
                str(row['adjusted_lex' + ref_tag]).split()[0].strip()
                if pd.notna(row['adjusted_lex' + ref_tag]) and len(str(row['adjusted_lex' + ref_tag]).split()) > 1
                else str(row['adjusted_lex' + ref_tag]).strip() if pd.notna(row['adjusted_lex' + ref_tag]) else ''
            ), 
            axis=1
        )

        if data == 'zaebuc':
            merged_df['pos_score'] = merged_df.apply(
                lambda row: pos_match_score_multi(row['ud' + input_tag], row['ud' + ref_tag]), axis=1
            )
        else:
            merged_df['pos_score'] = merged_df.apply(
                lambda row: pos_match_score_multi(row['pos' + input_tag], row['pos' + ref_tag]), axis=1
            )

        merged_df[['stemgloss_recall', 'stemgloss_precision', 'stemgloss_f1']] = merged_df.apply(
            lambda row: pd.Series(stemgloss_match_score_multi(row['stemgloss' + input_tag], row['stemgloss' + ref_tag])),
            axis=1
        )

        # Calculate sync score
        merged_df['sync_score'] = merged_df['lex_score'] + merged_df['pos_score'] + merged_df['stemgloss_f1']

        # Sort by sync_score and pos_lex_logprob to resolve ties
        merged_df.sort_values(by=['sentence_index', 'word_index', 'sync_score', 'pos_lex_logprob'], ascending=[True, True, False, False], inplace=True)

        # Select relevant columns

        if data == 'barec':
            result = merged_df[
                ['sentence_index', 'word_index', 'word' + ref_tag, 'lex' + input_tag, 'pos' + input_tag, 'stemgloss' + input_tag,  'adjusted_lex' + input_tag, 'lex' + ref_tag, 'adjusted_lex' + ref_tag, 'pos' + ref_tag, 'stemgloss' + ref_tag, 'lex_score', 'pos_score','stemgloss_f1', 'stemgloss_precision', 'stemgloss_recall', 'sync_score', 'pos_lex_logprob', 'lex_logprob', 'Comment', 'Status', 'Source', 'Book']
            ]
        elif data == 'zaebuc':
            result = merged_df[
                ['sentence_index', 'word_index', 'word' + ref_tag, 'lex' + input_tag, 'ud' + input_tag, 'pos', 'stemgloss' + input_tag,  'adjusted_lex' + input_tag, 'lex' + ref_tag, 'adjusted_lex' + ref_tag, 'ud' + ref_tag, 'stemgloss' + ref_tag, 'lex_score', 'pos_score','stemgloss_f1', 'stemgloss_precision', 'stemgloss_recall', 'sync_score', 'pos_lex_logprob', 'lex_logprob']
            ]
        else:
            result = merged_df[
                ['sentence_index', 'word_index', 'word' + ref_tag, 'lex' + input_tag, 'pos' + input_tag, 'stemgloss' + input_tag,  'adjusted_lex' + input_tag, 'lex' + ref_tag, 'adjusted_lex' + ref_tag, 'pos' + ref_tag, 'stemgloss' + ref_tag, 'lex_score', 'pos_score','stemgloss_f1', 'stemgloss_precision', 'stemgloss_recall', 'sync_score', 'pos_lex_logprob', 'lex_logprob']
            ]

    return result


##### FOR QURANIC DATA #######
# Function to clean the 'Word' column (Arabic letters and diacritics only)
def quran_clean_arabic_text(text):
    return re.sub(r'[^\u0621-\u064A\u064B-\u0652\u0671]', '', text)

def quran_clean_columns(df, word_column, lex_column):
    # Clean the 'Word' column
    df[word_column] = df[word_column].astype(str).apply(quran_clean_arabic_text)
    df[word_column] = df[word_column].str.replace('ءَا', 'آ', regex=False)
    df[word_column] = df[word_column].str.replace('ءا', 'آ', regex=False)
    df[word_column] = df[word_column].str.replace('ءأ', 'آ', regex=False)
    df[lex_column] = df[lex_column].astype(str).apply(quran_clean_arabic_text)
    return df

def fix_word_index_with_increment(df):
    df = df.copy()
    for sentence_idx in df['sentence_index'].unique():
        mask = df['sentence_index'] == sentence_idx
        temp_df = df[mask].copy()
        
        # Adjust word_index for the current sentence
        new_word_index = []
        current_index = -1  # Start before 0
        
        for idx in temp_df['word_index']:
            if idx == current_index:  # If same as the previous, keep it
                new_word_index.append(current_index)
            elif idx > current_index:  # Increment by 1 for non-repeated indices
                current_index += 1
                new_word_index.append(current_index)
        
        # Update the word_index in the original DataFrame
        df.loc[mask, 'word_index'] = new_word_index
    
    return df