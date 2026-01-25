from pathlib import Path
import hashlib

def hash_bytes(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def build_map(root):
    mapping = {}
    for path in root.rglob('*'):
        if path.is_file():
            rel = path.relative_to(root)
            mapping[str(rel).replace('\\', '/')] = (path.stat().st_size, hash_bytes(path))
    return mapping

old_root = Path('rust/temp_outputs/old')
new_root = Path('rust/temp_outputs/new')
old_map = build_map(old_root)
new_map = build_map(new_root)
old_set = set(old_map)
new_set = set(new_map)
only_old = sorted(old_set - new_set)
only_new = sorted(new_set - old_set)
print('missing in new (only in old):', only_old)
print('missing in old (only in new):', only_new)
common = sorted(old_set & new_set)
mismatch = []
for rel in common:
    old_size, old_hash = old_map[rel]
    new_size, new_hash = new_map[rel]
    if old_size != new_size or old_hash != new_hash:
        mismatch.append(rel)
print('mismatched files:', mismatch)
print('total files compared:', len(common))
