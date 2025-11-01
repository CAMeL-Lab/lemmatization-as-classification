# Lemmatization as Classification 

This repository contains the code and resources associated with the paper on **Lemmatization as a Classification Task: Results from Arabic across Multiple Genres**.  


# Requirements:
Here's how you can set up the environment using conda (assuming you have conda and cuda installed):


```bash
https://github.com/CAMeL-Lab/lemmatization-as-classification.git
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
