# run_all.py
import os
import sys
import argparse

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
sys.path.insert(0, REPO_ROOT)   # so we can import full_script and scripts/*


# Parse CLI arguments
parser = argparse.ArgumentParser(description="Run full disambiguation pipeline")
parser.add_argument('--datasets', nargs='+', default=['atb_dev'], help='List of dataset names')
parser.add_argument('--granularities', nargs='+', default=['lex_pos_stemgloss'], help='List of corresponding granularities')
parser.add_argument('--experiments', type=str, default='S2S,LexC+S2S', help='Comma-separated experiment names')
parser.add_argument('--morph_db',type=str,required=True,help='Path to the morphological database file')
args = parser.parse_args()

# Set up experiment config mapping
experiment_mapping = {
    'S2S': {'name': 'S2S', 'technique': 's2s', 'analyzer_set': 'all', 'tagger': False},
    'LexC+S2S': {'name': 'LexC+S2S', 'technique': 'LexC+S2S', 'analyzer_set': 'all', 'tagger': False},
    'All+Rand': {'name': 'All+Rand', 'technique': 'rand', 'analyzer_set': 'all', 'tagger': False},
    'Top+Rand': {'name': 'Top+Rand', 'technique': 'rand', 'analyzer_set': 'top', 'tagger': True},
    'Top+LogP': {'name': 'Top+LogP', 'technique': 'logp', 'analyzer_set': 'top', 'tagger': True},
    'Top+S2S+LogP': {'name': 'Top+S2S+LogP', 'technique': 's2s_logp', 'analyzer_set': 'top', 'tagger': True},
    'Top+LexC+LogP': {'name': 'Top+LexC+LogP', 'technique': 'lexc_logp', 'analyzer_set': 'top', 'tagger': True},
    'Top+LexC+S2S+LogP': {'name': 'Top+LexC+S2S+LogP', 'technique': 'lexc_s2s_logp', 'analyzer_set': 'top', 'tagger': True},
    'Top+Clust+LogP': {'name': 'Top+Clust+LogP', 'technique': 'clust_logp', 'analyzer_set': 'top', 'tagger': True},
    'Top+Clust+S2S+LogP': {'name': 'Top+Clust+S2S+LogP', 'technique': 'clust_s2s_logp', 'analyzer_set': 'top', 'tagger': True}
}

# Validate experiments
selected_experiments = [experiment_mapping[name] for name in args.experiments.split(',') if name in experiment_mapping]

# Set dynamic inputs as global variables used in full_script
globals()['dataset_names'] = args.datasets
globals()['granularities'] = args.granularities
globals()['experiments'] = selected_experiments
globals()['morph_db'] = args.morph_db

# Run full script
exec(open(os.path.join('run_pipeline.py')).read())
