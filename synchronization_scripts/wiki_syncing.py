import pandas as pd
from helpers import clean_words_with_camel_arclean, disambig_output, compute_highest_scoring_records_multi, normalize_text
from helpers import static_stuff_to_with_ours, bert_disambig, calima_analyzer


wiki_with_pos_df = pd.read_excel('original_datasets/wiki_news_dataset/WikiNews-26-06-2015-RefPOS.xlsx')
wiki_nizar_pos_map_df = pd.read_excel('original_datasets/wiki_news_dataset/WikiNews-26-06-2015-RefPOS+Nizar Map.xlsx')
wiki_nizar_pos_map_df['MADAMIRA POS'] = wiki_nizar_pos_map_df['MADAMIRA POS'].fillna("space")


wiki_df = pd.read_excel('original_datasets/wiki_news_dataset/WikiNews-26-06-2015-RefLemma.xlsx')
wiki_df['sentence_index'] = wiki_df['sentNo'].fillna(method='ffill').astype(int)
wiki_df['sentence_index'] = wiki_df['sentence_index'] - 1
wiki_df['word_index'] = wiki_df.groupby('sentence_index').cumcount()
wiki_df['word_index'] = wiki_df['word_index'].astype(int)

wiki_df.rename({"corrWord": "word", "refLemma": "lex"}, axis=1, inplace=True)
wiki_df[['refPOS(Farasa)', 'MADAMIRA POS' , 'Diff MADAMIRA POS']] = wiki_with_pos_df[['refPOS(Farasa)', 'MADAMIRA POS' , 'Diff MADAMIRA POS']]
wiki_df['Diff MADAMIRA POS'] = wiki_df['Diff MADAMIRA POS'].fillna("space")
wiki_df['MADAMIRA POS'] = wiki_df['MADAMIRA POS'].fillna("space")

###### Getting the WIKI POS MAPPING ######
wiki_mapped_pos = []
for i in range(0, len(wiki_df)):
    wiki_mapped_pos.append(wiki_nizar_pos_map_df[(wiki_nizar_pos_map_df['refPOS(Farasa)'] == wiki_df['refPOS(Farasa)'][i])
                        &(wiki_nizar_pos_map_df['MADAMIRA POS'] == wiki_df['MADAMIRA POS'][i])
                        &(wiki_nizar_pos_map_df['Diff MADAMIRA POS'] == wiki_df['Diff MADAMIRA POS'][i])].reset_index()['MADA'][0])

wiki_df['pos'] = wiki_mapped_pos
wiki_df['pos'] = wiki_df['pos'].str.split('/')


wiki_df = clean_words_with_camel_arclean(wiki_df, words_column="word")
for i in range(0, len(wiki_df)):
    if wiki_df['lex'][i] == '"':
        wiki_df['word'][i] = '"'
    elif wiki_df['lex'][i] == '،':
        wiki_df['lex'][i] = ','
    elif wiki_df['lex'][i] == '؟':
        wiki_df['lex'][i] = '?'


wiki_output_df = disambig_output(bert_disambig, calima_analyzer, wiki_df)

wiki_output_df.drop_duplicates(subset=['sentence_index', 'word_index', 'lex', 'pos', 'stemgloss'],inplace=True)
wiki_output_df.reset_index(drop=True)


# Apply normalization function and extract transformation counts into separate columns
wiki_output_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = wiki_output_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data='wiki_news')))
wiki_output_df.reset_index(drop=True, inplace=True)

wiki_df[['adjusted_lex', 'remove_fatha_before_alef', 'remove_damma_before_waw', 'remove_kasra_before_yaa',
        'remove_fatha_before_superscript_alef', 'replace_superscript_alef_with_fatha',
        'move_tanween_to_last_letter', 'remove_diacritics_last_letter',
        'replace_alef_wasla_kasra', 'fix_shadda_order', 'fix_sun_letter_shadda']] = wiki_df['lex'].apply(lambda x: pd.Series(normalize_text(x, data='wiki_news')))

### Getting all varities ###
wiki_df = wiki_df.explode('pos', ignore_index=True)


##### Compute the scores ####
sync_df = compute_highest_scoring_records_multi(wiki_df, wiki_output_df, lex_pos_flag=True, input_tag='_s31', ref_tag = '_wiki')
sync_df = sync_df.groupby(['sentence_index', 'word_index']).first().reset_index()

results = []
for (sentence_idx, word_idx), group in sync_df.groupby(['sentence_index', 'word_index']):
    if group['stemgloss'].isna().any():
        result = "UNK"
    if (group['sync_score'] == 0).all():
        result = "can't decide"    
    else:
        result = "done"
    results.append(result)


sync_df_filtered =  sync_df.groupby(['sentence_index', 'word_index']).first().reset_index()

## Editing the sync_status for some record based on some forced_conditions
sync_df_filtered['sync_status'] = results
for condition in static_stuff_to_with_ours:
    sync_df_filtered.loc[
        (sync_df_filtered['pos_s31'] == condition[0]) &
        (sync_df_filtered['lex_s31'] == condition[1]) &
        (sync_df_filtered['lex_wiki'] == condition[2]),
        'sync_status'
    ] = 'force_done'

gold_lex_list = []
gold_pos_list = []
gold_stemgloss_list = []
for i in range(0, len(sync_df_filtered)):

    if sync_df_filtered['sync_status'][i] == 'done':
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos_s31'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss'][i])
    
    if sync_df_filtered['sync_status'][i] == 'force_done':
        gold_lex_list.append(sync_df_filtered['lex_s31'][i])
        gold_pos_list.append(sync_df_filtered['pos_s31'][i])
        gold_stemgloss_list.append(sync_df_filtered['stemgloss'][i])

    elif sync_df_filtered['sync_status'][i] == "can't decide":
        gold_lex_list.append(sync_df_filtered['lex_wiki'][i])
        gold_pos_list.append(sync_df_filtered['pos_wiki'][i])
        gold_stemgloss_list.append('')

    elif sync_df_filtered['sync_status'][i] == "UNK":
        gold_lex_list.append(sync_df_filtered['lex_wiki'][i])
        gold_pos_list.append(sync_df_filtered['pos_wiki'][i])
        gold_stemgloss_list.append('')
        
sync_df_filtered.insert(3, 'gold_lex', gold_lex_list)
sync_df_filtered.insert(3, 'gold_pos', gold_pos_list)
sync_df_filtered.insert(3, 'gold_stemgloss', gold_stemgloss_list)

sync_df_filtered.to_csv("synced_wiki_data.csv", index=False)