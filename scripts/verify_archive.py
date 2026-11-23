"""Verify the distributed files against SHA256SUMS.txt (standard library only)."""
from pathlib import Path
from hashlib import sha256
import sys

root = Path(__file__).resolve().parents[1]
failures = []
count = 0
for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    if not line.strip() or line.startswith('#'):
        continue
    expected, relative = line.split(maxsplit=1)
    relative = relative.lstrip('*')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append(f'Missing or invalid path: {relative}')
        continue
    if sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(f'Checksum mismatch: {relative}')
    count += 1
if failures:
    print('\n'.join(failures))
    sys.exit(1)
print(f'PASS: {count} files match SHA256SUMS.txt.')
