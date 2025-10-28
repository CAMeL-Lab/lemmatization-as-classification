from tqdm import tqdm
import pandas as pd
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator


from .modeling import rename_and_prepare_input

# === Load the S31 Morphological Database with Encoded LPG and Cluster Assignments ===
s31_db_path = 'source/clustered_s31_DB/latest_full_s31_in_2000_cluster_lex_pos_stemgloss.csv'
final_s31_lex = pd.read_csv(s31_db_path)

# === Load the S31 Morphological Database with Encoded LPG and Cluster Assignments ===
s31_db_path = 'source/clustered_s31_DB/latest_full_s31_in_2000_cluster_lex_pos_stemgloss.csv'
final_s31_lex = pd.read_csv(s31_db_path)

# Rename relevant columns for clarity and consistency
final_s31_lex.rename(columns={
    'lex_encoded_from_dict': 'LPG_encoded',
    'completed_clusters': 'clusters'
}, inplace=True)

# === Create Mapping from LPG String to Encoded Class and Cluster ID ===
LPG_to_encoded_class = final_s31_lex.set_index('lex_pos_stemgloss')[['LPG_encoded', 'clusters']].to_dict()


### Loading the disambiguator with calime-s31 DB and NOAN_PROP Backoff
back_off = 'NOAN_PROP'
db = MorphologyDB('source/Morphological_DB/calima-msa-s31_0.4.2.utf8.db')
calima_analyzer = Analyzer(db, back_off)
bert_disambig = BERTUnfactoredDisambiguator.pretrained('msa', top = 5000, pretrained_cache=False)
bert_disambig._analyzer = calima_analyzer

# === Select the Best Disambiguation Option Based on Classification or Clustering ===
def select_top_disambiguation(df, disambig_df, s2s_df, use_s2s=False, classification = False, clustering=False):
    if 'sync_status' in df:
        sync_status_list = df['sync_status'].copy()
    else:
        sync_status_list = []

    if use_s2s:
        s2s_lookup = s2s_df.set_index(['sentence_index', 'word_index'])['predicted_lemma'].to_dict()
    else:
        s2s_lookup = {}

    disambig_mi = disambig_df.set_index(['sentence_index', 'word_index'])

    filtered_rows, failed_indices = [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering rows"):
        key = (row['sentence_index'], row['word_index'])
        if classification == True:
            top1 = row['top1_lex_info']
            if not isinstance(top1, list) or not top1 or not isinstance(top1[0], dict):
                raise ValueError("Invalid top1_lex_info format")

            target_lex = top1[0].get('Lex')
            target_pos = top1[0].get('POS')
            target_stemgloss = top1[0].get('stemgloss')

        elif clustering == True:
            target_cluster = int(row['predicted_clusters'])
            
        s2s_lemma = s2s_lookup.get(key, "").strip() if use_s2s else None

        group_df = disambig_mi.loc[key] if key in disambig_mi.index else pd.DataFrame()
        group_df = group_df.to_frame().T if isinstance(group_df, pd.Series) else group_df

        if not group_df.empty:
            if classification == True:

                first_pos = group_df.iloc[0].get('pos', '')
                if first_pos in {'punc', 'digit'}:
                    filtered_rows.append(group_df.iloc[0])
                    continue  # move to the next word

                else:
                    full_match = group_df[
                        (group_df['lex'] == target_lex) &
                        (group_df['pos'] == target_pos) &
                        (group_df['stemgloss'] == target_stemgloss)
                    ]
                    if not full_match.empty:
                        if use_s2s == True:
                            s2s_match = full_match[full_match['lex'] == s2s_lemma]
                            filtered_rows.append(s2s_match.iloc[0] if not s2s_match.empty else full_match.iloc[0])
                        else:
                            filtered_rows.append(full_match.iloc[0])

                    else:
                        if use_s2s == True:
                            s2s_only_match = group_df[group_df['lex'] == s2s_lemma]
                            filtered_rows.append(s2s_only_match.iloc[0] if not s2s_only_match.empty else group_df.iloc[0])
                        else:
                            filtered_rows.append(group_df.iloc[0])

            elif clustering == True:

                first_pos = group_df.iloc[0].get('pos', '')
                if first_pos in {'punc', 'digit'}:
                    filtered_rows.append(group_df.iloc[0])
                    continue  # move to the next word
                
                else:
                    full_match = group_df[
                        (group_df['clusters'] == int(target_cluster))
                    ]
                    if not full_match.empty:
                        if use_s2s == True:
                            s2s_match = full_match[full_match['lex'] == s2s_lemma]
                            filtered_rows.append(s2s_match.iloc[0] if not s2s_match.empty else full_match.iloc[0])
                        else:
                            filtered_rows.append(full_match.iloc[0])
                    else:
                        if use_s2s == True:
                            s2s_only_match = group_df[group_df['lex'] == s2s_lemma]
                            filtered_rows.append(s2s_only_match.iloc[0] if not s2s_only_match.empty else group_df.iloc[0])
                        else:
                            filtered_rows.append(group_df.iloc[0])
                
        else:
            failed_indices.append(key)
            print(f"No record found at {key}")
    filtered_rows = pd.DataFrame(filtered_rows)
    if len(sync_status_list)>0:
        filtered_rows['sync_status'] = sync_status_list
    return filtered_rows, pd.DataFrame(failed_indices, columns=['sentence_index', 'word_index'])

def get_evaluatble_data(df, dataset_type='atb'):
    df = df.copy()

    if (dataset_type == 'atb_test') or (dataset_type == 'atb_dev'):
        df_gold_dev2 = df[
            (df['lex_num'] != 0) |
            ((df['lex_num'] == 0) & (df['gold_pos'].isin(['punc', 'digit'])))
        ]
    
    elif dataset_type == 'barec':
        df_gold_dev2 = df[
            (df['sync_status'] != 'not ready') &
            (df['sync_status'] != 'UNK-No Gold') &
            (df['sync_status'] != 'No Gold')
        ]
    
    elif dataset_type == 'nemlar':
        df_gold_dev2 = df[
            (df['sync_status'] != 'UNK-No Gold') &
            (df['sync_status'] != 'No Gold')
        ]

    elif dataset_type in ['wiki', 'quran', 'zaebuc']:
        df_gold_dev2 = df.copy()
        df_gold_dev2 = df_gold_dev2.loc[:, ~df_gold_dev2.columns.duplicated()]

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    return df_gold_dev2

def merge_with_gold(df, final_df, data='', eval='lex'):
    final_df = final_df.reset_index(drop=True)
    if eval == 'lex':
        final_df['gold_lex'] = df['gold_lex']
    if eval == 'lex_pos':
        final_df['gold_lex'] = df['gold_lex']
        final_df['gold_pos'] = df['gold_pos']
    elif eval == 'lex_pos_stemgloss':
        final_df['gold_lex'] = df['gold_lex']
        final_df['gold_pos'] = df['gold_pos']
        final_df['gold_stemgloss'] = df['gold_stemgloss']
    
    try:
        final_df['lex_num'] = df['lex_num']
    except:
        pass

    final_df2 = get_evaluatble_data(final_df, dataset_type=data)

    results = {}
    if eval == 'lex':
        results['lex_accuracy'] = (final_df2['gold_lex'] == final_df2['lex']).mean() * 100
    elif eval == 'lex_pos':
        results['lex_accuracy'] = (final_df2['gold_lex'] == final_df2['lex']).mean() * 100
        results['lex_pos_accuracy'] = ((final_df2['gold_lex'] == final_df2['lex']) & (final_df2['gold_pos'] == final_df2['pos'])).mean() * 100
    elif eval == 'lex_pos_stemgloss':
        results['lex_accuracy'] = (final_df2['gold_lex'] == final_df2['lex']).mean() * 100
        results['lex_pos_accuracy'] = ((final_df2['gold_lex'] == final_df2['lex']) & (final_df2['gold_pos'] == final_df2['pos'])).mean() * 100
        results['lex_pos_stemgloss_accuracy'] = ((final_df2['gold_lex'] == final_df2['lex']) & 
                                                  (final_df2['gold_pos'] == final_df2['pos']) & 
                                                  (final_df2['gold_stemgloss'] == final_df2['stemgloss'])).mean() * 100

    return final_df, results

def evaluate_disambiguation_with_sentences(
    df,
    s2s_df,
    data_name,
    word_column='word',
    sentence_column_name='sentence_index',
    word_column_name='word_index',
    technique='logp',  # 'logp', 'rand', or 's2s_logp'
    granularity='lex',  # 'lex', 'lex_pos', or 'lex_pos_stemgloss'
    analyzer_set='top',  # 'top' or 'all'
    tagger=True  # True to use the tagger, False to use only the analyzer
):
    """
    Evaluate disambiguation performance using a given selection technique and output granularity.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the words and gold standard annotations.

    data_name : str
        A string identifier used for naming the output CSV file.

    word_column : str, default='word'
        The name of the column in `df` that contains the word tokens.

    sentence_column_name : str, default='sentence_index'
        The name of the column in `df` that contains the sentence index for each word.

    word_column_name : str, default='word_index'
        The name of the column in `df` that contains the word index within each sentence.

    technique : str, default='logP'
        The disambiguation selection methodology to apply. Options include:
        - 'logp': Selects the top candidate based on the highest `pos_lex_logprob`.
        - 'rand': Applies deterministic random selection based on `original_index`.
        - 's2s_logp': First selects based on logp, then filters using the S2S model's predicted lemma.

    granularity : str, default='lex'
        The level of granularity used for evaluation. Options are:
        - 'lex'       : Evaluate using lexical match only (L).
        - 'lex_pos'   : Evaluate using lexical and POS match (LP).
        - 'lex_pos_stemgloss' : Evaluate using lexical, POS, and gloss match (LPG).

    analyzer_set : str, default='top'
        Defines which analyses to consider from the disambiguator/analyzer.
        - 'top': Use only top-scoring analyses (score == 1).
        - 'all': Use all analyses returned.

    tagger : bool, default=True
        Whether to use the tagger for disambiguation.
        - True : Use the BERT-based tagger (e.g., CALIMA BERT).
        - False: Use only the morphological analyzer (e.g., CALIMA analyzer).

    Returns:
    -------
    top1_df : pd.DataFrame
        DataFrame containing the selected (top-1) disambiguation result per word.

    disambig_df : pd.DataFrame
        DataFrame containing all analyses or disambiguation results prior to selection.
    """


    # === Step 1: Prepare list of tokenized sentences and their original indices ===
    sentences_list = []
    indices_list = []

    if technique == 's2s':
        results = {}
        df['lex'] = s2s_df['predicted_lemma']
        df.loc[df['gold_pos'] == 'punc', 'lex'] = df['word']
        df.loc[df['gold_pos'] == 'digit', 'lex'] = df['word']

        df = get_evaluatble_data(df, dataset_type=data_name)
        # === Compute metrics based on granularity ===
        results['lex_accuracy'] = (df['lex'] == df['gold_lex']).mean() * 100

        return df, results
    
    if technique == 'LexC+S2S':
        results = {}
        lexc_s2s_df = pd.read_csv(f'/Users/mms10094/Downloads/{data_name}_LexC_seq2seq_loop_until_found.csv')
        
        lexc_s2s_df.rename(columns={
            'lex': 'gold_lex',
            'pos': 'gold_pos',
            'stemgloss': 'gold_stemgloss'
        }, inplace=True)
        
        lexc_s2s_df = lexc_s2s_df.loc[:, ~lexc_s2s_df.columns.duplicated()]

        lexc_s2s_df.loc[lexc_s2s_df['gold_pos'] == 'punc', 'final_lex'] = lexc_s2s_df['word']
        lexc_s2s_df.loc[lexc_s2s_df['gold_pos'] == 'digit', 'final_lex'] = lexc_s2s_df['word']
        df['lex'] = lexc_s2s_df['final_lex']
        df['pos'] = lexc_s2s_df['final_pos']
        df['stemgloss'] = lexc_s2s_df['final_stemgloss']
        df = get_evaluatble_data(df, dataset_type=data_name)
        # === Compute metrics based on granularity ===
        if granularity in ['lex', 'lex_pos', 'lex_pos_stemgloss']:
            results['lex_accuracy'] = (df['lex'] == df['gold_lex']).mean() * 100

        if granularity in ['lex_pos', 'lex_pos_stemgloss']:
            results['lex_pos_accuracy'] = ((df['lex'] == df['gold_lex']) & (df['pos'] == df['gold_pos'])).mean() * 100

        if granularity == 'lex_pos_stemgloss':
            results['lex_pos_stemgloss_accuracy'] = (
                (df['lex'] == df['gold_lex']) &
                (df['pos'] == df['gold_pos']) &
                (df['stemgloss'] == df['gold_stemgloss'])
            ).mean() * 100

        return df, results

    grouped = df.groupby(sentence_column_name)
    for _, group in grouped:
        group = group.sort_values(word_column_name)
        sentences_list.append(list(group[word_column]))
        indices_list.append(group.index.tolist())

    # === Step 2: Analyze or Disambiguate ===
    if not tagger:
        records = []
        for i, sentence in tqdm(enumerate(sentences_list), total=len(sentences_list), desc="Analyzing sentences"):
            word_analyses = calima_analyzer.analyze_words(sentence)
            indices = indices_list[i]

            for j, word_analysis in enumerate(word_analyses):
                word = sentence[j]
                original_idx = indices[j]

                for analysis in word_analysis.analyses:
                    record = {
                        'original_index': original_idx,
                        'sentence_index': i,
                        'word_index': j,
                        'word': word
                    }
                    record.update(analysis)
                    records.append(record)

        disambig_df = pd.DataFrame(records)

    else:
        results = bert_disambig.disambiguate_sentences(sentences_list)
        flattened_analyses = []

        for sentence_idx, sentence in enumerate(results):
            indices = indices_list[sentence_idx]

            for word_idx, word in enumerate(sentence):
                word_text = word.word
                original_idx = indices[word_idx]

                for analysis in word.analyses:
                    entry = {
                        'sentence_index': sentence_idx,
                        'word_index': word_idx,
                        'word': word_text,
                        'original_index': original_idx,
                        'diac': analysis.diac,
                        'score': analysis.score
                    }
                    entry.update(analysis.analysis)
                    flattened_analyses.append(entry)

        disambig_df = pd.DataFrame(flattened_analyses)

    # === Step 3: Candidate Selection ===
    disambig_df = disambig_df.drop_duplicates(
        subset=['sentence_index', 'word_index', 'lex', 'pos', 'stemgloss']
    ).reset_index(drop=True)

    if analyzer_set == 'top':
        disambig_df = disambig_df[disambig_df['score'] == 1].reset_index(drop=True)
        
    if technique == 'logp':
        if analyzer_set =='all':
            disambig_df = disambig_df.sort_values(
                by=['sentence_index', 'word_index', 'pos_lex_logprob'],
                ascending=[True, True, False]
            ).reset_index(drop=True)

        final_df = disambig_df.drop_duplicates(subset=['sentence_index', 'word_index'], keep='first').reset_index(drop=True)
        if 'sync_status' in df.columns:
            final_df['sync_status'] = df['sync_status']
        
    elif technique == 's2s_logp':
        s2s_lookup = df.set_index(['sentence_index', 'word_index'])['s2s_predicted_lemma'].to_dict()
        selected_rows = []

        for (s_idx, w_idx), group in disambig_df.groupby(['sentence_index', 'word_index']):
            predicted_lemma = s2s_lookup.get((s_idx, w_idx), '').strip()
            match_idx = group[group['lex'] == predicted_lemma].index

            selected = group.loc[match_idx[0]] if not match_idx.empty else group.iloc[0]
            selected_rows.append(selected)

        final_df = pd.DataFrame(selected_rows).reset_index(drop=True)
        if 'sync_status' in df.columns:
            final_df['sync_status'] = df['sync_status']

    elif technique == 'rand':
        selected_rows = []
        for (s_idx, w_idx), group in disambig_df.groupby(['sentence_index', 'word_index']):
            original_idx = group.iloc[0]['original_index']
            pos = original_idx % len(group)
            selected_rows.append(group.iloc[pos])

        final_df = pd.DataFrame(selected_rows).reset_index(drop=True)
        if 'sync_status' in df.columns:
            final_df['sync_status'] = df['sync_status']

    elif technique == 'lexc_logp':
        print("Step 1: Preparing input and S2S predictions...")
        df = rename_and_prepare_input(df, final_s31_lex, classification= True, clustering=False)
        
        for i in tqdm(range(0, len(df))):
            try:
                if (str(df['predicted_LPG_top1'][i]) == '1'):
                    df['s2s_predicted_lemma'][i] = df['word'][i]
                    df['top1_lex_info'][i][0]['Lex'] = df['word'][i]
                    df['top1_lex_info'][i][0]['POS'] = df['gold_pos'][i]
                    df['top1_lex_info'][i][0]['stemgloss'] = df['word'][i]
            except:
                pass
        
        disambig_df["lex_pos_stemgloss"] = (
            disambig_df["lex"] + "_" + disambig_df["pos"] + "_" + disambig_df["stemgloss"]
        )
        disambig_df["encoded_lemma"] = disambig_df["lex_pos_stemgloss"].map(lambda x: LPG_to_encoded_class['LPG_encoded'].get(x))
        disambig_df["clusters"] = disambig_df["lex_pos_stemgloss"].map(lambda x: LPG_to_encoded_class['clusters'].get(x))

        # Assign 1 only if 'pos' is digit or punc
        disambig_df.loc[
            disambig_df['pos'].isin(['digit', 'punc']),
            ['encoded_lemma', 'clusters']
        ] = [1, 1]

        print("Step 2: Filtering top matches...")
        final_df, failed_df = select_top_disambiguation(df, disambig_df, s2s_df, use_s2s=False, classification= True, clustering = False)
        final_df = final_df.reset_index(drop=True)
        if 'sync_status' in df.columns:
            final_df['sync_status'] = df['sync_status']
        
    elif technique == 'lexc_s2s_logp':
        print("Step 1: Preparing input and S2S predictions...")
        df = rename_and_prepare_input(df, final_s31_lex, classification= True, clustering=False)
        
        for i in tqdm(range(0, len(df))):
            try:
                if (str(df['predicted_LPG_top1'][i]) == '1'):
                    df['s2s_predicted_lemma'][i] = df['word'][i]
                    df['top1_lex_info'][i][0]['Lex'] = df['word'][i]
                    df['top1_lex_info'][i][0]['POS'] = df['gold_pos'][i]
                    df['top1_lex_info'][i][0]['stemgloss'] = df['word'][i]
            except:
                pass
        
        disambig_df["lex_pos_stemgloss"] = (
            disambig_df["lex"] + "_" + disambig_df["pos"] + "_" + disambig_df["stemgloss"]
        )
        disambig_df["encoded_lemma"] = disambig_df["lex_pos_stemgloss"].map(lambda x: LPG_to_encoded_class['LPG_encoded'].get(x))
        disambig_df["clusters"] = disambig_df["lex_pos_stemgloss"].map(lambda x: LPG_to_encoded_class['clusters'].get(x))

        # Assign 1 only if 'pos' is digit or punc
        disambig_df.loc[
            disambig_df['pos'].isin(['digit', 'punc']),
            ['encoded_lemma', 'clusters']
        ] = [1, 1]

        print("Step 3: Filtering top matches...")
        final_df, failed_df = select_top_disambiguation(df, disambig_df, s2s_df, use_s2s=True, classification= True, clustering = False)
        final_df = final_df.reset_index(drop=True)
        if 'sync_status' in df.columns:
            final_df['sync_status'] = df['sync_status']
       
    elif technique == 'clust_logp':
        print("Step 1: Preparing input and S2S predictions...")
        df = rename_and_prepare_input(df, final_s31_lex, classification= False, clustering=True)
        
        for i in tqdm(range(0, len(df))):
            try:
                if (df['predicted_clusters'][i] == '1'):
                    df['s2s_predicted_lemma'][i] = df['word'][i]
                    df['top1_lex_info'][i][0]['Lex'] = df['word'][i]
                    df['top1_lex_info'][i][0]['POS'] = df['gold_pos'][i]
                    df['top1_lex_info'][i][0]['stemgloss'] = df['word'][i]
            except:
                pass
        
        df['clusters'] = df['predicted_clusters']

        disambig_df["lex_pos_stemgloss"] = (
            disambig_df["lex"] + "_" + disambig_df["pos"] + "_" + disambig_df["stemgloss"]
        )
        disambig_df["encoded_lemma"] = disambig_df["lex_pos_stemgloss"].map(lambda x: LPG_to_encoded_class['LPG_encoded'].get(x))
        disambig_df["clusters"] = disambig_df["lex_pos_stemgloss"].map(lambda x: LPG_to_encoded_class['clusters'].get(x))

        # Assign 1 only if 'pos' is digit or punc
        disambig_df.loc[
            disambig_df['pos'].isin(['digit', 'punc']),
            ['encoded_lemma', 'clusters']
        ] = [1, 1]

        print("Step 3: Filtering top matches...")
        final_df, failed_df = select_top_disambiguation(df, disambig_df, s2s_df, use_s2s=False, classification= False, clustering = True)
        final_df = final_df.reset_index(drop=True)
        if 'sync_status' in df.columns:
            final_df['sync_status'] = df['sync_status']
        # final_df = merge_with_gold(df, final_df, data = data_name, eval=granularity)

    elif technique == 'clust_s2s_logp':
        print("Step 1: Preparing input and S2S predictions...")
        df = rename_and_prepare_input(df, final_s31_lex, classification= False, clustering=True)

        for i in tqdm(range(0, len(df))):
            try:
                if (df['predicted_clusters'][i] == '1'):
                    df['s2s_predicted_lemma'][i] = df['word'][i]
                    df['top1_lex_info'][i][0]['Lex'] = df['word'][i]
                    df['top1_lex_info'][i][0]['POS'] = df['gold_pos'][i]
                    df['top1_lex_info'][i][0]['stemgloss'] = df['word'][i]
            except:
                pass
        
        df['clusters'] = df['predicted_clusters']

        disambig_df["lex_pos_stemgloss"] = (
            disambig_df["lex"] + "_" + disambig_df["pos"] + "_" + disambig_df["stemgloss"]
        )
        disambig_df["encoded_lemma"] = disambig_df["lex_pos_stemgloss"].map(lambda x: LPG_to_encoded_class['LPG_encoded'].get(x))
        disambig_df["clusters"] = disambig_df["lex_pos_stemgloss"].map(lambda x: LPG_to_encoded_class['clusters'].get(x))

        # Assign 1 only if 'pos' is digit or punc
        disambig_df.loc[
            disambig_df['pos'].isin(['digit', 'punc']),
            ['encoded_lemma', 'clusters']
        ] = [1, 1]

        print("Step 3: Filtering top matches...")
        final_df, failed_df = select_top_disambiguation(df, disambig_df, s2s_df, use_s2s=True, classification= False, clustering = True)
        final_df = final_df.reset_index(drop=True)
        if 'sync_status' in df.columns:
            final_df['sync_status']  = df['sync_status']

    else:
        raise ValueError(f"Unsupported technique: {technique}")

    final_df, results = merge_with_gold(df, final_df, data = data_name, eval=granularity)

    
    return final_df, results
