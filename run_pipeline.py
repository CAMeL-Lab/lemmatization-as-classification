import pandas as pd
import re
from scripts.evaluation import evaluate_disambiguation_with_sentences


accuracy_to_tag = {
    'lex_accuracy': 'L',
    'lex_pos_accuracy': 'LP',
    'lex_pos_stemgloss_accuracy': 'LPG'
}
# === Final result structure
table_rows = []

for idx, data_name in enumerate(dataset_names):
    granularity = granularities[idx]
    print(f"\n=== Processing dataset: {data_name} with granularity: {granularity} ===")

    # === Load and preprocess your data ===
    use_s2s = any(exp['technique'] in ['s2s', 's2s_logp', 'lexc_s2s_logp', 'clust_s2s_logp']
              for exp in experiments)
              
    # s2s_df = pd.read_csv(f'data/S2S Output Files/{data_name}_predictions.csv')
    if use_s2s:
        print("→ Loading S2S predictions...")
        s2s_df = pd.read_csv(f"data/S2S Output Files/{data_name}_predictions.csv")
        s2s_df['pos'] = df['pos']
        s2s_df['word_index'] = df['word_index']
        s2s_df['sentence_index'] = df['sentence_index']
        s2s_df.loc[s2s_df['pos'] == 'punc', 'predicted_lemma'] = s2s_df['word']
        s2s_df.loc[s2s_df['pos'] == 'digit', 'predicted_lemma'] = s2s_df['word']
        df['s2s_predicted_lemma'] = s2s_df['predicted_lemma']
    else:
        print("→ S2S not selected — skipping S2S predictions.")
        s2s_df = pd.DataFrame()

    df = pd.read_csv(f'data/Synced Datasets/{data_name} data.csv')

    if 'gold_pos' in df.columns:
        df.rename(columns={'gold_pos': 'pos'}, inplace=True)

    df = df.loc[:, ~df.columns.duplicated()]

    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0653\u0654\u0655]')
    df['word'] = df['word'].astype(str).apply(lambda x: arabic_diacritics.sub('', x))

    df['original_index'] = df.index
    index_lookup = df.set_index(['sentence_index', 'word_index'])['original_index'].to_dict()

    df.rename(columns={
        'lex': 'gold_lex',
        'pos': 'gold_pos',
        'stemgloss': 'gold_stemgloss'
    }, inplace=True)

    for config in experiments:
        print(f"  → Running experiment: {config['name']}")
        final_df, metrics= evaluate_disambiguation_with_sentences(
            df=df,
            s2s_df = s2s_df,
            morph_db = morph_db,
            data_name=data_name,
            word_column='word',
            granularity=granularity,
            technique=config['technique'],
            analyzer_set=config['analyzer_set'],
            tagger=config['tagger']
        )

        for acc_type, tag in accuracy_to_tag.items():
            # Skip metrics not relevant to this dataset's granularity
            if acc_type == 'lex_accuracy' or \
               (acc_type == 'lex_pos_accuracy' and granularity in ['lex_pos', 'lex_pos_stemgloss']) or \
               (acc_type == 'lex_pos_stemgloss_accuracy' and granularity == 'lex_pos_stemgloss'):

                # Find or create the row for (dataset, tag)
                row = next((r for r in table_rows if r['Dataset'] == data_name and r['Tag'] == tag), None)
                if not row:
                    row = {'Dataset': data_name, 'Tag': tag}
                    table_rows.append(row)

                # Fill the accuracy value for this technique
                row[config['name']] = metrics.get(acc_type, '-')

    print(pd.DataFrame(table_rows))
# === Convert to DataFrame
structured_df = pd.DataFrame(table_rows)

# === Reorder columns
ordered_columns = ['Dataset', 'Tag'] + [exp['name'] for exp in experiments]
structured_df = structured_df[ordered_columns]

# === Save to CSV or print
structured_df.to_csv("structured_accuracy_table_nemlar.csv", index=False)
print("\n✅ Structured table saved to structured_accuracy_table.csv")
print(structured_df)