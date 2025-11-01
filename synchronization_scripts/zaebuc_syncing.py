import pandas as pd
from helpers import clean_words_with_camel_arclean, disambig_output, compute_highest_scoring_records_multi, normalize_text
from helpers import static_stuff_to_with_ours, ud_mada_pos_mapping, bert_disambig, calima_analyzer

#### READING and CLEANING data ####
file_path = "../data/Original Datasets/ZAEBUC Data/AR-all.extracted.corrected.analyzed.corrected-FINAL.tsv"
zabuc_df = pd.read_csv(file_path, sep='\\t')
zabuc_df = zabuc_df.dropna(subset=['Word'])

unique_documents = zabuc_df['Document'].unique()
sentence_indices = []
word_indices = []

# Loop through each unique document to assign sentence_index and word_index
for sentence_idx, document in enumerate(unique_documents):
    # Filter rows for the current document
    document_df = zabuc_df[zabuc_df['Document'] == document]
    
    # Get the word indices for each row in the document
    for word_idx in range(len(document_df)):
        sentence_indices.append(sentence_idx)
        word_indices.append(word_idx) 

# Add the new columns to the DataFrame
zabuc_df['sentence_index'] = sentence_indices
zabuc_df['word_index'] = word_indices
zabuc_df = zabuc_df.rename({'Word':'word' , 'Manual_Diacritized_Lemma': 'lex', 'Manual_POS' : "ud", "Gloss": 'stemgloss'}, axis=1)
zabuc_df.reset_index(inplace=True, drop=True)
zabuc_df = clean_words_with_camel_arclean(zabuc_df, words_column="word")
for i in range(0, len(zabuc_df)):
    if zabuc_df['ud'][i] == 'PUNCT':
        zabuc_df['lex'][i] = zabuc_df['stemgloss'][i]
        zabuc_df['word'][i] = zabuc_df['stemgloss'][i]

    if zabuc_df['lex'][i] == '٪':
        zabuc_df['lex'][i] = '%'

## RUNING DISAMBIG IF NOT EXIST, AND IF EXIST THEN LOAD IT ##
zabuc_output_df = disambig_output(bert_disambig, calima_analyzer, zabuc_df)
zabuc_output_df.drop_duplicates(subset=['sentence_index', 'word_index', 'lex', 'pos', 'stemgloss'], inplace=True)

### Normalizing lex diacs ###
zabuc_output_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = zabuc_output_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data='zaebuc')))
zabuc_output_df.reset_index(drop=True, inplace=True)

zabuc_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = zabuc_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data='zaebuc')))

### Running the SYNC PROCESS AND GETTING ALL SCORES ###
sync_df = compute_highest_scoring_records_multi(zabuc_df, zabuc_output_df, lex_pos_stemgloss_flag=True, input_tag='_s31', ref_tag = '_zaebuc', data='zaebuc')

results = []
for (sentence_idx, word_idx), group in sync_df.groupby(['sentence_index', 'word_index']):
    if group['stemgloss_zaebuc'].isna().any():
        result = "UNK"
    elif (group['sync_score'] == 0).all():
        result = "can't decide"    
    else:
        result = "done"
    results.append(result)

sync_df_filtered = sync_df.groupby(['sentence_index', 'word_index']).first().reset_index()
sync_df_filtered['sync_status'] = results

### Forcing some sync status ###
for condition in static_stuff_to_with_ours:
    sync_df_filtered.loc[
        (sync_df_filtered['pos'] == condition[0]) &
        (sync_df_filtered['lex_s31'] == condition[1]) &
        (sync_df_filtered['lex_zaebuc'] == condition[2]),
        'sync_status'
    ] = 'force_done'

gold_lex_list = []
gold_pos_list = []
gold_stemgloss_list = []
for i in range(0, len(sync_df_filtered)):

    if sync_df_filtered['sync_status'][i] == 'done':
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_s31'][i])

    elif (sync_df_filtered['sync_status'][i] == "can't decide") & ((sync_df_filtered['ud_zaebuc'][i] == 'ADP+PART') | (sync_df_filtered['ud_zaebuc'][i] == 'CCONJ+ADP+PART')):
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_s31'][i])
        sync_df_filtered['sync_status'][i] = 'force_done'

    elif (sync_df_filtered['sync_status'][i] == "UNK") & (sync_df_filtered['ud_zaebuc'][i] == 'PART+ADP+PART'):
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_s31'][i])
        sync_df_filtered['sync_status'][i] = 'force_done'

    elif sync_df_filtered['sync_status'][i] == 'force_done':
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_s31'][i])

    elif sync_df_filtered['sync_status'][i] == "can't decide":
        gold_lex_list.append(sync_df_filtered['lex_zaebuc'][i])
        gold_pos_list.append(ud_mada_pos_mapping.get(sync_df_filtered['ud_zaebuc'][i], sync_df_filtered['ud_zaebuc'][i]))
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_zaebuc'][i])

    elif sync_df_filtered['sync_status'][i] == "UNK":
        gold_lex_list.append(sync_df_filtered['lex_zaebuc'][i])
        gold_pos_list.append(ud_mada_pos_mapping.get(sync_df_filtered['ud_zaebuc'][i], sync_df_filtered['ud_zaebuc'][i]))
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_zaebuc'][i])
        
sync_df_filtered.insert(3, 'gold_lex', gold_lex_list)
sync_df_filtered.insert(3, 'gold_pos', gold_pos_list)
sync_df_filtered.insert(3, 'gold_stemgloss', gold_stemgloss_list)

sync_df_filtered = sync_df_filtered[['sentence_index', 'word_index', 'word_zaebuc', 'gold_lex', 'gold_pos', 'gold_stemgloss']]
sync_df_filtered.rename({'word_zaebuc' : 'word'}, axis=1, inplace=True)
sync_df_filtered.to_csv("../data/Synced Datasets/zaebuc data.csv", index=False)