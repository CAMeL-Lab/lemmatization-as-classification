import pandas as pd
import re
from helpers import fix_sentence_and_word_index, disambig_output, compute_highest_scoring_records_multi, normalize_text
from helpers import static_stuff_to_with_ours, static_dict, bert_disambig, calima_analyzer

# Process all XML files and get the data
nemlar_df = pd.read_csv('../data/Original Datasets/Nemlar Data/nemlar_data.csv')
# Sort and reset index
nemlar_df = nemlar_df.sort_values(by=['sentence_index', 'word_index']).reset_index(drop=True)

# Adjust sentence index and handle empty word
nemlar_df.loc[nemlar_df['word_index'] == -1, 'word'] = ' '
nemlar_df = nemlar_df[nemlar_df['word'] != '']
nemlar_df = nemlar_df[nemlar_df['word'] != ' '].reset_index(drop=True)
nemlar_df = nemlar_df[nemlar_df['word'] != '  '].reset_index(drop=True)
nemlar_df = nemlar_df[nemlar_df['word'] != '   '].reset_index(drop=True)
nemlar_df = nemlar_df[nemlar_df['word'] != '    '].reset_index(drop=True)
nemlar_df = nemlar_df[nemlar_df['word'] != '\t'].reset_index(drop=True)
nemlar_df = fix_sentence_and_word_index(nemlar_df)

nemlar_df.rename({"lemma": "lex"}, axis=1, inplace=True)

# Replace specific lex values
for i in range(len(nemlar_df)):
    if nemlar_df.loc[i, 'lex'] == '"':
        nemlar_df.loc[i, 'word'] = '"'
    elif nemlar_df.loc[i, 'lex'] == '،':
        nemlar_df.loc[i, 'lex'] = ','

only_diacritics_pattern = re.compile(r'^[\u064B-\u0652]+$')
nemlar_df = nemlar_df[~nemlar_df['word'].str.match(only_diacritics_pattern, na=False)]
nemlar_df = fix_sentence_and_word_index(nemlar_df)

nemlar_output_df = disambig_output(bert_disambig, calima_analyzer, nemlar_df)

# nemlar_output_df.to_csv('../EMNLP_Output_files/synced_data/nemlar_disambig_NONE.csv', index=False)
# nemlar_output_df = pd.read_csv('../EMNLP_Output_files/disambig_files/nemlar_disambig_NONE.csv')
nemlar_output_df.drop_duplicates(subset=['sentence_index', 'word_index', 'word', 'lex', 'pos', 'stemgloss'],inplace=True)
nemlar_output_df.reset_index(drop=True, inplace=True)


# Split and explode the `lex` column
nemlar_df['lex'] = nemlar_df['lex'].str.split()
nemlar_df = nemlar_df.explode('lex', ignore_index=True)
nemlar_df['lex'] = nemlar_df['lex'].apply(lambda x: x.strip() if isinstance(x, str) else x)
nemlar_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = nemlar_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data='nemlar')))

nemlar_output_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = nemlar_output_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data='nemlar')))

### Getting the Syncing score ###
result_df = compute_highest_scoring_records_multi(nemlar_df, nemlar_output_df, lex_flag=True, input_tag='_s31', ref_tag='_nemlar')

results = []
for (sentence_idx, word_idx), group in result_df.groupby(['sentence_index', 'word_index']):
    if group['stemgloss'].isna().any():
        result = "UNK"
    elif (group['sync_score'] == 0).all():
        result = "can't decide"    
    else:
        result = "done"
    results.append(result)

result_df = result_df.groupby(['sentence_index', 'word_index']).first().reset_index()
result_df['sync_status'] = results

### Forcing some sync_status ###
result_df.loc[
    (result_df['sync_status'] == "can't decide") & (result_df['lex_nemlar'] == '+'),
    'sync_status'
] = 'No Gold'
result_df.loc[
    (result_df['sync_status'] == "can't decide") & (result_df['lex_nemlar'] == '-'),
    'sync_status'
] = 'No Gold'

result_df.loc[
    (result_df['sync_status'] == "UNK") & (result_df['lex_nemlar'] == '+'),
    'sync_status'
] = 'UNK-No Gold'
result_df.loc[
    (result_df['sync_status'] == "UNK") & (result_df['lex_nemlar'] == '-'),
    'sync_status'
] = 'UNK-No Gold'

for condition in static_stuff_to_with_ours:
    result_df.loc[
        (result_df['pos'] == condition[0]) &
        (result_df['lex_s31'] == condition[1]) &
        (result_df['lex_nemlar'] == condition[2]),
        'sync_status'
    ] = 'force_done'

gold_lex_list = []
gold_pos_list = []
gold_stemgloss_list = []
for i in range(0, len(result_df)):

    if result_df['sync_status'][i] == 'done':
        gold_lex_list.append(result_df['lex_s31'][i])
    
    elif result_df['sync_status'][i] == 'force_done':
        gold_lex_list.append(result_df['lex_s31'][i])

    elif result_df['sync_status'][i] == 'No Gold':
        gold_lex_list.append('nan')

    elif result_df['sync_status'][i] == "can't decide":
        gold_lex_list.append(result_df['lex_nemlar'][i])

    elif result_df['sync_status'][i] == "UNK":
        gold_lex_list.append(result_df['lex_nemlar'][i])
    
    elif result_df['sync_status'][i] == "UNK-No Gold":
        gold_lex_list.append('nan')
        
result_df.insert(3, 'gold_lex', gold_lex_list)

# Loop through the DataFrame and update the columns if the word matches
for i in range(len(result_df)):
    word = result_df.at[i, 'word_nemlar']
    if word in static_dict:
        # Get the tuple values
        lex, pos, stemgloss = static_dict[word]
        # Update the columns
        result_df.at[i, 'gold_lex'] = lex
        result_df.at[i, 'sync_status'] = 'force_done'

result_df = result_df[['sentence_index', 'word_index', 'word_nemlar', 'gold_lex']]
result_df.rename({"word_nemlar":"word"}, inplace=True, axis=1)
result_df.to_csv("../data/Synced Datasets/nemlar data.csv", index=False)