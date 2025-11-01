import pandas as pd
from helpers import clean_words_with_camel_arclean, fix_sentence_and_word_index, disambig_output, compute_highest_scoring_records_multi, normalize_text
from helpers import static_stuff_to_with_ours, static_dict, bert_disambig, calima_analyzer

# Reading BAREC Data Analysis
barec_df_categories = pd.read_csv('../data/Original Datasets/Barec Data/Lemma-Annotation-BAREC-P1-sources - Each Sentence Source & Book.csv')

# Reading BAREC Data
barec_df = pd.read_csv('../data/Original Datasets/Barec Data/Barec Data.csv')

barec_df_categories.rename({"Sentence ID" : "Sentence Index"}, axis=1, inplace=True)
barec_df = barec_df.merge(barec_df_categories, on=['Batch Number', 'Sentence Index'], how='left')
barec_df = clean_words_with_camel_arclean(barec_df, words_column="Word")


### CLEANING AND STRUCTURING DATA ###
barec_df['Chosen Output'] = barec_df['Chosen Output'].fillna('')
barec_df[['lex', 'pos', 'gloss']] = barec_df['Chosen Output'].str.split('---', expand=True)
barec_df.drop('Chosen Output', axis=1, inplace=True)
barec_df

# Double Checking that all puncs are correct
barec_df.loc[barec_df['Word'] == '-', ['lex', 'gloss']] = '-'
barec_df.loc[barec_df['Word'] == '-', ['pos']] = 'punc'
condition = (barec_df['lex'] == '') & barec_df['gloss'].notna() & (barec_df['pos'] == 'punc')
barec_df.loc[condition, 'lex'] = barec_df['gloss']
barec_df.loc[condition, 'Word'] = barec_df['gloss']
barec_df.loc[barec_df['pos'] == 'punc', ['Word', 'gloss']] = barec_df['lex']
barec_df.loc[barec_df['pos'] == 'digit', ['Word', 'gloss']] = barec_df['lex']

# Starting from index 0
barec_df.rename({"gloss": 'stemgloss', 'Sentence Index': "sentence_index", "Word Index": "word_index"}, axis=1, inplace=True)
barec_df['sentence_index'] = barec_df['sentence_index']-1
barec_df['word_index'] = barec_df['word_index']-1
barec_df.rename({"Word" : "word"}, axis=1, inplace=True)
# barec_df = barec_df.dropna(subset=['word']).reset_index(drop=True)
barec_df = barec_df[barec_df['word'] != '']
barec_df = fix_sentence_and_word_index(barec_df)

barec_df['word'] = barec_df['word'].astype(str)

# if there is no saved disambiguator output
barec_output_df = disambig_output(bert_disambig, calima_analyzer, barec_df)
barec_output_df.drop_duplicates(subset=['sentence_index', 'word_index', 'lex', 'pos', 'stemgloss'],inplace=True)
barec_output_df.reset_index(drop=True, inplace=True)

### Doing the lex diac normalization ###
barec_output_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = barec_output_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data='barec')))
barec_output_df.reset_index(drop=True, inplace=True)

barec_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = barec_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data='barec')))


### getting th sync scores ###
sync_df = compute_highest_scoring_records_multi(barec_df, barec_output_df, lex_pos_stemgloss_flag=True, input_tag='_s31', ref_tag = '_CM', data='barec')

results = []
for (sentence_idx, word_idx), group in sync_df.groupby(['sentence_index', 'word_index']):
    if group['stemgloss_s31'].isna().any():
        result = "UNK"
    elif (group['sync_score'] == 0).all():
        result = "can't decide"    
    else:
        result = "done"
    results.append(result)

sync_df_filtered = sync_df.groupby(['sentence_index', 'word_index']).first().reset_index()
sync_df_filtered['sync_status'] = results


### Adjusting som sync_status ###
sync_df_filtered.loc[
    (sync_df_filtered['sync_status'] == "UNK") & (sync_df_filtered['lex_CM'] == ''),
    'sync_status'
] = 'UNK-No Gold'
sync_df_filtered

for condition in static_stuff_to_with_ours:
    sync_df_filtered.loc[
        (sync_df_filtered['pos_s31'] == condition[0]) &
        (sync_df_filtered['lex_s31'] == condition[1]) &
        (sync_df_filtered['lex_CM'] == condition[2]),
        'sync_status'
    ] = 'force_done'

sync_df_filtered.loc[
    (sync_df_filtered['sync_status'] == "can't decide") & (sync_df_filtered['lex_CM'] == ''),
    'sync_status'
] = 'No Gold'


sync_df_filtered.rename({"word_CM" : 'word'}, inplace=True, axis=1)
gold_lex_list = []
gold_pos_list = []
gold_stemgloss_list = []
for i in range(0, len(sync_df_filtered)):

    if sync_df_filtered['sync_status'][i] == 'done':
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos_s31'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_s31'][i])

    elif sync_df_filtered['sync_status'][i] == 'force_done':
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_s31'][i])

    elif sync_df_filtered['sync_status'][i] == "can't decide":
        gold_lex_list.append(sync_df_filtered['lex_CM'][i])
        gold_pos_list.append(sync_df_filtered['pos_CM'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_CM'][i])

    elif sync_df_filtered['sync_status'][i] == "UNK":
        gold_lex_list.append(sync_df_filtered['lex_CM'][i])
        gold_pos_list.append(sync_df_filtered['pos_CM'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_CM'][i])

    elif sync_df_filtered['sync_status'][i] == "UNK-No Gold":
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos_s31'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_s31'][i])
    
    elif sync_df_filtered['sync_status'][i] == "No Gold":
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos_s31'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss_s31'][i])
        
sync_df_filtered.insert(3, 'gold_lex', gold_lex_list)
sync_df_filtered.insert(3, 'gold_pos', gold_pos_list)
sync_df_filtered.insert(3, 'gold_stemgloss', gold_stemgloss_list)

for i in range(len(sync_df_filtered)):
    word = sync_df_filtered.at[i, 'word']
    if word in static_dict:
        # Get the tuple values
        lex, pos, stemgloss = static_dict[word]
        # Update the columns
        sync_df_filtered.at[i, 'gold_lex'] = lex
        sync_df_filtered.at[i, 'gold_pos'] = pos
        sync_df_filtered.at[i, 'gold_stemgloss'] = stemgloss
        sync_df_filtered.at[i, 'sync_status'] = 'force_done'

sync_df_filtered = sync_df_filtered[['sentence_index', 'word_index', 'word', 'gold_lex', 'gold_pos', 'gold_stemgloss']]
sync_df_filtered = fix_sentence_and_word_index(sync_df_filtered)
sync_df_filtered.to_csv('../data/Synced Datasets/barec data.csv', index=False)