import argparse
import pandas as pd
from scripts.evaluation import evaluate_disambiguation_with_sentences


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate disambiguation with sentences.")
    parser.add_argument("--morph_db", type=str, required=True, help="Path to morphological database.")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the input CSV file.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ====== Fixed parameters ======
    word_column = "word"
    granularity = ""
    technique = "clust_logp"  # lex, lex_pos, lex_pos_stemgloss
    analyzer_set = "top"
    tagger = True
    s2s_df_path = "s2s_output.csv"  # fixed default file

    print("Loading data...")
    df = pd.read_csv(args.df_path)
    # s2s_df = pd.read_csv(s2s_df_path)

    print("Running evaluation...")
    final_df, metrics = evaluate_disambiguation_with_sentences(
        df=df,
        s2s_df='',
        morph_db=args.morph_db,
        data_name=args.data_name,
        word_column=word_column,
        granularity=granularity,
        technique=technique,
        analyzer_set=analyzer_set,
        tagger=tagger
    )

    print("\n✅ Evaluation Done")
    print("\n📌 Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    final_df_out = f"{args.data_name}_results.csv"
    final_df[['word', 'lex', 'pos', 'stemgloss']].to_csv(final_df_out, index=False)
    print(f"📂 Results saved → {final_df_out}")


if __name__ == "__main__":
    main()
