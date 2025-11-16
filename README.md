# Lemmatization as Classification 

This repository contains the code and resources associated with the paper on [**Lemmatization as a Classification Task: Results from Arabic across Multiple Genres**.  ](https://aclanthology.org/2025.emnlp-main.1525/)


# Requirements:
Here's how you can set up the environment using conda (assuming you have conda and cuda installed):


```bash
git clone https://github.com/CAMeL-Lab/lemmatization-as-classification.git
cd lemmatization-as-classification

conda create -n lemma_class_env python=3.12.8
conda activate lemma_class_env

pip install -r requirements.txt
```

# Repository Structure:
1- [data](https://github.com/CAMeL-Lab/lemmatization-as-classification/tree/main/data/):
  - [Original Datasets](https://github.com/CAMeL-Lab/lemmatization-as-classification/tree/main/data/Original%20Datasets) : All raw datasets collected from their respective official sources.
  - [Synced Datasets](https://github.com/CAMeL-Lab/lemmatization-as-classification/tree/main/data/Synced%20Datasets) : synchronized versions of the datasets generated through our synchronization process (as described in the paper).
  - [S2S Output Files](https://github.com/CAMeL-Lab/lemmatization-as-classification/tree/main/data/S2S%20Output%20Files) : Outputs from the Seq2Seq lemmatization models for each dataset.

2- [Morphological Databases](https://github.com/CAMeL-Lab/lemmatization-as-classification/tree/main/Morphological%20Databases): Contains the SAMA S31 muddled morphological database used across all experiments.

3- [LPG_clusters_and_classes](https://github.com/CAMeL-Lab/lemmatization-as-classification/tree/main/LPG_clusters_and_classes): Cluster and class assignments for each lemma–pos–gloss triplet, as introduced and analyzed in the paper.

4- [synchronization_scripts](https://github.com/CAMeL-Lab/lemmatization-as-classification/tree/main/synchronization_scripts) : Scripts required to transform the original datasets into their synchronized version following the methodology described in the paper.

5- [scripts](https://github.com/CAMeL-Lab/lemmatization-as-classification/tree/main/scripts) : All experiment execution scripts used for lemmatization, disambiguation, and evaluation in the paper.

6- `main.py` and `run_pipeline.py`: the core scripts for executing all experiments presented in the paper.

# Experiments and Reproducibility:

Run the main script to reproduce the results presented in the paper using the configurable parameters below:

```sh
python main.py \
  --datasets <INPUT_TXT_PATH> \
  --granularities <INPUT_VARIANT> \
  --experiments <TECHNIQUES>
  --morph_db <MORPHOLOGY_DATABASE> \
```

| Parameter         | Available Options                                                        | Description                                              |
| ----------------- | ------------------------------------------------------------------------ | -------------------------------------------------------- |
| `datasets`      | `atb_dev`, `atb_test`, `barec`, `nemlar`, `wiki`, `quran`, `zaebuc`  | Choose one or multiple datasets to run experiments on    |
| `granularities` | `lex`, `lex_pos`, `lex_pos_stemgloss`                                    | Select level of morphological granularity for prediction |
| `experiments`   | `S2S`, `LexC+S2S`, `All+Rand`, `Top+Rand`, `Top+LogP`, `Top+S2S+LogP`, `Top+LexC+LogP`, `Top+LexC+S2S+LogP`, `Top+Clust+LogP`, `Top+Clust+S2S+LogP`                                                | Choose which lemmatization technique(s) to run           |

## Preparing the Morphological Database and LPG Cluster/Class Mappings
To prepare the required MSA Morphological Database, you must download the source LDC package: [Standard Arabic Morphological Analyzer (SAMA) v3.1 database](https://catalog.ldc.upenn.edu/LDC2010L01).
Once downloaded, generate the CALIMA-MSA S31 database using ```muddler```:
```sh
muddler unmuddle \
  -s "/path/to/LDC2010L01.tgz" \
  -m "Morphological Databases/Sama_s31.muddle" \
  "Morphological Databases/calima-msa-s31_0.4.2.utf8.db"
```
This command produces the CALIMA-MSA S31 morphological database used across the pipeline.

#### Preparing the LPG Cluster/Class Mapping
Using the same SAMA v3.1 source file, generate the LPG cluster–class mapping by running:
```sh
muddler unmuddle \
  -s "/path/to/LDC2010L01.tgz" \
  -m "LPG_clusters_and_classes/LPG_clusters_classes.muddle" \
  "LPG_clusters_and_classes/LPG_clusters.csv" 
```
This will create the LPG clusters and class mapping file required for experiments involving LPG-based features.

## Preparing the Penn Arabic Treebank (if needed)

All datasets used n the paper are already included in the repo, except for the Penn Arabic Treebank (PATB), which must be built locally due to LDC licensing restrictions.
- [LDC2010T13](https://catalog.ldc.upenn.edu/LDC2010T13): ```atb1_v4_1_LDC2010T13.tgz```
- [LDC2011T09](https://catalog.ldc.upenn.edu/LDC2011T09): ```atb_2_3.1_LDC2011T09.tgz```
- [LDC2010T08](https://catalog.ldc.upenn.edu/LDC2010T08): ```atb3_v3_2_LDC2010T08.tgz```

1. Extract all three archives.
2. Combine their contents into a single folder.
3. Compress the folder again into one archive with name ```MSA_atb123.zip```.
4. Then generate the PATB CSV data by running:
```sh
muddler unmuddle \
  -s “/path/to/MSA atb123.zip" \
  -m "data/Synced Datasets/atb_test data.muddle" \
  "data/Synced Datasets/atb_test data.csv”
```

## Single dataset, one granularity, one technique
```bash
python main.py \
  --datasets atb_dev \
  --granularities lex_pos_stemgloss \
  --experiments Top+LogP
```

## Multiple datasets with corresponding granularities and techniques
```bash
python main.py \
  --datasets zaebuc barec atb_test quran \
  --granularities lex_pos_stemgloss lex_pos lex_pos_stemgloss lex \
  --experiments Top+LogP,Top+Clust+S2S+LogP
```

## Running the Best MSA Lemmatizer (Disambiguation + Clustering)
To run the **```best-performing MSA lemmatizer```**, which combines the ```MSA Disambiguator``` with the ```LPG clustering technique```, execute the following command:
```bash
python run_best_msa_lemmatizer.py \
    --morph_db "Morphological Databases/calima-msa-s31_0.4.2.utf8.db" \
    --data_name "wiki" \
    --df_path "data/Synced Datasets/wiki data.csv"
```

| Parameter     | Example Value                                          | Description                                                                                     |
| ------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `--morph_db`  | `Morphological Databases/calima-msa-s31_0.4.2.utf8.db` | Path to the compiled **CALIMA-MSA S31** morphological database used by the disambiguator.       |
| `--data_name` | `wiki`, `atb_dev`, `custom_name`               | A label used **only for saving the output file** generated by this lemmatization run.           |
| `--df_path`   | `data/Synced Datasets/wiki data.csv`                   | Path to the input CSV file containing **sentence_index**, **word_index**, and **word** columns. |

This command will run the full MSA lemmatization pipeline and save the results under the specified data_name.

## Citation

If you find this repository, the models, or the datasets useful in your research, please consider citing:
```bibtex
@inproceedings{saeed-habash-2025-lemmatization,
    title = "Lemmatization as a Classification Task: Results from {A}rabic across Multiple Genres",
    author = "Saeed, Mostafa  and
      Habash, Nizar",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    abstract = "Lemmatization is crucial for NLP tasks in morphologically rich languages with ambiguous orthography like Arabic, but existing tools face challenges due to inconsistent standards and limited genre coverage. This paper introduces two novel approaches that frame lemmatization as classification into a Lemma-POS-Gloss (LPG) tagset, leveraging machine translation and semantic clustering. We also present a new Arabic lemmatization test set covering diverse genres, standardized alongside existing datasets. We evaluate character-level sequence-to-sequence models, which perform competitively and offer complementary value, but are limited to lemma prediction (not LPG) and prone to hallucinating implausible forms. Our results show that classification and clustering yield more robust, interpretable outputs, setting new benchmarks for Arabic lemmatization."
}
```
