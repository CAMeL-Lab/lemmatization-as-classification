import numpy as np
import torch
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json 
from .disambig_utils import map_lpg_info, chunk_sentence

# === Load ID-to-Cluster Mapping for Model Outputs ===
with open('source/clustered_s31_DB/id_to_cluster.json', encoding='utf-8') as f:
    id2cluster = json.load(f)
id2cluster = {int(k): v for k, v in id2cluster.items()}

# === Load ID-to-LPG Label Mapping for Model Outputs ===
with open('source/clustered_s31_DB/id_to_lex_class.json', encoding='utf-8') as f:
    id2LPGlabel = json.load(f)
id2LPGlabel = {int(k): v for k, v in id2LPGlabel.items()}

# === Load Fine-tuned BERT Model for LPG Classification ===
classif_output_dir = "source/fine-tuned-models/classification/bert_base_arabic_camelbert_msa_pos_msa_lex_pos_stemgloss_with_UNK_relabeled"
lex_model = AutoModelForTokenClassification.from_pretrained(classif_output_dir)
lex_tokenizer = AutoTokenizer.from_pretrained(classif_output_dir)

# === Load Fine-tuned BERT Model for LPG Clustering ===
clust_output_dir = "source/fine-tuned-models/clustering/bert_base_arabic_camelbert_msa_pos_2000_clusters_lex_pos_stemgloss_with_UNK_relabeled"
clustering_model = AutoModelForTokenClassification.from_pretrained(clust_output_dir)
clustering_tokenizer = AutoTokenizer.from_pretrained(clust_output_dir)


# === Model Inference for a Single Chunk ===
def get_predictions_for_chunk(chunk, tokenizer, model):
    """
    Tokenizes and runs a model on a chunk of words.

    Args:
        chunk (list): List of tokens.
        tokenizer: HuggingFace tokenizer.
        model: Token classification model.

    Returns:
        logits: Raw output logits.
        hidden_states: Last hidden layer representation.
        word_ids: Mapping from tokens to word indices.
    """
    inputs = tokenizer(
        chunk,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

    return logits, hidden_states, inputs.word_ids(batch_index=0)

# === Align Predictions with Original Words ===
def align_predictions_and_embeddings(logits, hidden_states, word_ids, id_to_cluster):
    """
    Aligns model output logits with original word positions and returns predicted clusters.

    Args:
        logits (torch.Tensor): Model output logits.
        hidden_states (torch.Tensor): Last hidden states (not used here).
        word_ids (list): Token-to-word index mapping.
        id_to_cluster (dict): Mapping from label ID to cluster string.

    Returns:
        list: Cluster predictions for each word.
    """
    word_logits = {}

    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            word_logits.setdefault(word_id, []).append(logits[0, i, :].cpu().numpy())

    predictions = []
    for word_id in sorted(word_logits.keys()):
        avg_logits = np.mean(word_logits[word_id], axis=0)
        probs = softmax(avg_logits)
        pred_label = np.argmax(probs)
        predictions.append(id_to_cluster[pred_label])

    return predictions

# === Process Full DataFrame of Sentences ===
def process_sentences(df, tokenizer, model, id_to_cluster):
    """
    Runs token classification on each sentence in a DataFrame, handling long sentences via chunking.

    Args:
        df (pd.DataFrame): Must contain 'sentence_index' and 'word' columns.
        tokenizer: HuggingFace tokenizer.
        model: Token classification model.
        id_to_cluster (dict): Label ID to cluster string mapping.

    Returns:
        list: Aggregated predictions for all words in all sentences.
    """
    all_predictions = []

    sentence_indices = df['sentence_index'].unique()

    for idx in tqdm(sentence_indices, desc="Processing Sentences"):
        words = df[df['sentence_index'] == idx]['word'].tolist()

        # Check tokenized length
        tokenized = tokenizer(words, is_split_into_words=True, truncation=True, padding=True)
        if len(tokenized["input_ids"]) > 512:
            chunks = chunk_sentence(words, tokenizer, max_length=512)
        else:
            chunks = [words]

        for chunk in chunks:
            logits, hidden_states, word_ids = get_predictions_for_chunk(chunk, tokenizer, model)
            chunk_preds = align_predictions_and_embeddings(logits, hidden_states, word_ids, id_to_cluster)

            # Debug: Length mismatch
            if len(chunk_preds) != len(chunk):
                print(f"Chunk mismatch:\nPredictions: {len(chunk_preds)}\nWords: {len(chunk)}\nChunk: {chunk}\n---")

            all_predictions.extend(chunk_preds)

    return all_predictions

# === Apply Token Classification Models (Classification or Clustering) ===
def rename_and_prepare_input(df, final_s31_lex, classification=False, clustering=False):
    """
    Applies the appropriate model (classification or clustering) to the input DataFrame
    and returns the modified DataFrame with predictions added.

    Args:
        df (pd.DataFrame): Input DataFrame with at least a 'word' column.
        classification (bool): If True, apply LPG classification model.
        clustering (bool): If True, apply clustering model.

    Returns:
        pd.DataFrame: DataFrame with additional prediction columns.
    """
    # Apply classification model
    if classification:
        df['predicted_LPG_top1'] = process_sentences(
            df=df,
            tokenizer=lex_tokenizer,
            model=lex_model,
            id_to_cluster=id2LPGlabel
        )
        df['top1_lex_info'] = df['predicted_LPG_top1'].apply(lambda x: map_lpg_info([x], final_s31_lex))

    # Apply clustering model
    elif clustering:
        df['predicted_clusters'] = process_sentences(
            df=df,
            tokenizer=clustering_tokenizer,
            model=clustering_model,
            id_to_cluster=id2cluster
        )

    return df
