import pandas as pd
import re
from helpers import clean_words_with_camel_arclean, disambig_output, compute_highest_scoring_records_multi, normalize_text, lex_transliterate, quran_clean_columns, fix_word_index_with_increment
from helpers import static_stuff_to_with_ours, static_dict, bert_disambig, calima_analyzer

file_path = 'original_datasets/quran_orig_dataset/fullquran_analysis_magold5(prep_fixd).txt'

# Open the file and read its contents into the text variable
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Split the text into lines for processing
lines = text.splitlines()

# Initialize storage for results
analysis_results = []
current_word = None
current_number = None
current_lex = "UNK"
current_kais_pos = "UNK"  # Store KAIS POS
current_analysis_pos = "UNK"  # Store ANALYSIS POS
processing_analysis = False  # Track if inside the ANALYSIS section

for line in lines:
    # Identify new word section
    if line.startswith(";; WORD"):
        current_word = line.split()[2]  # Extract the word
        current_lex = "UNK"
        current_kais_pos = "UNK"
        current_analysis_pos = "UNK"
        processing_analysis = False  # Reset when a new word starts
    
    # Extract KAIS lex and pos
    elif line.startswith(";; KAIS") and current_word:
        number_match = re.search(r"\((\d+:\d+:\d+:\d+)\)", line)
        lex_match = re.search(r"LEM:([^|]+)", line)
        pos_match = re.search(r"POS:([^|]+)", line)
        
        if number_match:
            current_number = number_match.group(1)
        if lex_match:
            current_lex = lex_match.group(1).strip()
        if pos_match:
            current_kais_pos = pos_match.group(1).strip()  # Store KAIS POS
    
    # Detect start of ANALYSIS section
    elif line.startswith(";;; ANALYSIS"):
        processing_analysis = True  # Set flag to start extracting analysis
    
    # Extract lex and pos from ANALYSIS section if available
    elif processing_analysis and (line.startswith("*1.0") or line.startswith("_1.0")):
        lex_match = re.search(r"lex:([^\s]+)", line)
        pos_match = re.search(r"pos:([^\s]+)", line)
        
        if not lex_match:
            lex_match = re.search(r"lex:UNK", line)
        if not pos_match:
            pos_match = re.search(r"pos:UNK", line)
        
        if current_word and current_number:
            lex_value = lex_match.group(1).strip() if lex_match else current_lex
            current_analysis_pos = pos_match.group(1).strip() if pos_match else "UNK"  # Store ANALYSIS POS
            analysis_results.append((current_word, current_number, lex_value, current_kais_pos, current_analysis_pos))

# Convert Analysis results to DataFrame
analysis_df = pd.DataFrame(analysis_results, columns=["Word", "Number", "Analysis_LEX", "KAIS_POS", "Analysis_POS"])
analysis_df['Analysis_LEX'] = analysis_df['Analysis_LEX'].astype(str)   
analysis_df['Analysis_LEX'] = analysis_df.apply(lambda row: lex_transliterate(row['Analysis_LEX']), axis=1)
analysis_df['Word'] = analysis_df['Word'].astype(str)   
analysis_df['Word'] = analysis_df.apply(lambda row: lex_transliterate(row['Word']), axis=1)

# Assign sentence index
sentence_index_column = -1
sentence_indices = []
prev_second_number = None

for _, row in analysis_df.iterrows():
    second_number = row['Number'].split(':')[1]
    if second_number != prev_second_number:
        sentence_index_column += 1
    sentence_indices.append(sentence_index_column)
    prev_second_number = second_number

analysis_df['sentence_index'] = sentence_indices
analysis_df = clean_words_with_camel_arclean(analysis_df, words_column="Word")
analysis_df = quran_clean_columns(analysis_df, 'Word', 'Analysis_LEX')
analysis_df['original_word'] = analysis_df['Word']


# for analysis purposes
beeb = analysis_df.__deepcopy__()
beeb.drop_duplicates(subset=['Word', 'Number'], inplace=True)
kais_pos= beeb['KAIS_POS'].reset_index(drop=True)
original_word= beeb['original_word'].reset_index(drop=True)
quranic_df = analysis_df
quranic_df[['sora_index', 'bla2', 'word_index', "bla"]] = quranic_df['Number'].str.split(':', expand=True)
quranic_df['sora_index'] = quranic_df['sora_index'].astype(int)
quranic_df['bla2'] = quranic_df['bla2'].astype(int)
quranic_df['word_index'] = quranic_df['word_index'].astype(int)
quranic_df['word_index'] = quranic_df['word_index'] - 1
quranic_df.drop(['bla', 'bla2', 'sora_index'], axis=1, inplace=True)
quranic_df['Word'] = quranic_df['Word'].str.replace('+·', '', regex=False)


quranic_df.rename({'Word' : 'word'}, axis=1, inplace=True)
quranic_df.rename({
    "Analysis_LEX": "LEM",
    "Analysis_POS": "POS"
}, inplace=True, axis=1)

quranic_df = fix_word_index_with_increment(quranic_df)
quranic_df.rename({
    "LEM": "lex",
    "POS": "pos"
}, inplace=True, axis=1)
quranic_df

intg_df = quranic_df[quranic_df['KAIS_POS'] == 'INTG']
duplicated_df = intg_df.copy()
duplicated_df['pos'] = 'pron_interrog'
quranic_df = pd.concat([quranic_df, duplicated_df], ignore_index=True)
quranic_df = quranic_df.sort_values(by=['sentence_index', 'word_index']).reset_index(drop=True)
temp_df = quranic_df.drop_duplicates(subset=['word', 'sentence_index', 'word_index', 'Number'])

quranic_output_df = disambig_output(bert_disambig, calima_analyzer, temp_df)
quranic_output_df.drop_duplicates(subset=['sentence_index', 'word_index', 'lex', 'pos', 'stemgloss', 'pos_lex_logprob', 'lex_logprob'],inplace=True)
quranic_output_df.reset_index(drop=True)

quranic_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = quranic_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data = 'quranic')))


quranic_output_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = quranic_output_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data = 'quranic')))

sync_df = compute_highest_scoring_records_multi(quranic_df, quranic_output_df, lex_pos_flag=True, input_tag='_s31', ref_tag='_quranic')

results = []
for (sentence_idx, word_idx), group in sync_df.groupby(['sentence_index', 'word_index']):
    if group['stemgloss'].isna().any():
        result = "UNK"
    elif (group['sync_score'] == 0).all():
        result = "can't decide"    
    else:
        result = "done"
    results.append(result)


sync_df_filtered = sync_df.groupby(['sentence_index', 'word_index']).first().reset_index()
sync_df_filtered['sync_status'] = results
sync_df_filtered.loc[sync_df_filtered['lex_quranic'] == 'UNK', 'sync_status'] = 'No Gold'
kais_pos = kais_pos.reset_index(drop=True)
sync_df_filtered['Kais_pos'] = kais_pos

for condition in static_stuff_to_with_ours:
    sync_df_filtered.loc[
        (sync_df_filtered['pos_s31'] == condition[0]) &
        (sync_df_filtered['lex_s31'] == condition[1]) &
        (sync_df_filtered['lex_quranic'] == condition[2]),
        'sync_status'
    ] = 'force_done'

sync_df_filtered['original_word'] = original_word

gold_lex_list = []
gold_pos_list = []
gold_stemgloss_list = []
for i in range(0, len(sync_df_filtered)):

    if sync_df_filtered['sync_status'][i] == 'done':
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos_s31'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss'][i])
    
    elif sync_df_filtered['sync_status'][i] == 'force_done':
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos_s31'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss'][i])

    elif (sync_df_filtered['sync_status'][i] == 'No Gold') & (sync_df_filtered['pos_s31'][i] == 'pron'):
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos_s31'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss'][i])
        sync_df_filtered['sync_status'][i] = 'force_done'

    elif (sync_df_filtered['sync_status'][i] == 'No Gold') & (sync_df_filtered['Kais_pos'][i] == 'INL'):
        gold_lex_list.append(sync_df_filtered['word_quranic'][i])
        gold_pos_list.append(sync_df_filtered['pos_quranic'][i])
        gold_stemgloss_list.append('')
        sync_df_filtered['sync_status'][i] = 'force_done'

    elif sync_df_filtered['sync_status'][i] == 'No Gold':
        gold_lex_list.append('nan')
        gold_pos_list.append('nan')
        gold_stemgloss_list.append('nan')

    elif sync_df_filtered['sync_status'][i] == "can't decide":
        gold_lex_list.append(sync_df_filtered['lex_quranic'][i])
        gold_pos_list.append(sync_df_filtered['pos_quranic'][i])
        gold_stemgloss_list.append('')

    elif sync_df_filtered['sync_status'][i] == "UNK":
        gold_lex_list.append(sync_df_filtered['lex_quranic'][i])
        gold_pos_list.append(sync_df_filtered['pos_quranic'][i])
        gold_stemgloss_list.append('')
        
sync_df_filtered.insert(3, 'gold_lex', gold_lex_list)
sync_df_filtered.insert(3, 'gold_pos', gold_pos_list)
sync_df_filtered.insert(3, 'gold_stemgloss', gold_stemgloss_list)

# Loop through the DataFrame and update the columns if the word matches
for i in range(len(sync_df_filtered)):
    word = sync_df_filtered.at[i, 'original_word']
    if word in static_dict:
        # Get the tuple values
        lex, pos, stemgloss = static_dict[word]
        # Update the columns
        sync_df_filtered.at[i, 'gold_lex'] = lex
        sync_df_filtered.at[i, 'gold_pos'] = pos
        sync_df_filtered.at[i, 'gold_stemgloss'] = stemgloss
        sync_df_filtered.at[i, 'sync_status'] = 'force_done'

kais_pos = kais_pos.reset_index(drop=True)
sync_df_filtered['Kais_pos'] = kais_pos

sync_df_filtered.to_csv("synced_quran_data.csv", index=False)