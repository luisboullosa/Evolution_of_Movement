from pathlib import Path
import difflib
files = [
    ('config.json', Path('rust/temp_outputs/old/config.json'), Path('rust/temp_outputs/new/config.json')),
    ('frames.tsv', Path('rust/temp_outputs/old/frames.tsv'), Path('rust/temp_outputs/new/frames.tsv')),
    ('nn_index.json', Path('rust/temp_outputs/old/nn_index.json'), Path('rust/temp_outputs/new/nn_index.json')),
    ('run_config.json', Path('rust/temp_outputs/old/run_config.json'), Path('rust/temp_outputs/new/run_config.json')),
]
for name, old_path, new_path in files:
    old_text = old_path.read_text().splitlines()
    new_text = new_path.read_text().splitlines()
    diff = list(difflib.unified_diff(old_text, new_text, fromfile=f'old/{name}', tofile=f'new/{name}', lineterm=''))
    print(f"\n=== {name} differences (showing first 10 lines) ===")
    for line in diff[:10]:
        print(line)
    if len(diff) > 10:
        print('...')
