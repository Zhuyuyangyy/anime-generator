#!/usr/bin/env python3
"""Strip WSL noise junk: write clean version to /tmp first, then cp to /mnt/d/"""
import os, shutil, re

BASE = '/mnt/d/ZYY Project/anime-generator/'

def find_code_start(raw):
    for p in [b'"""', b"'''", b'\nimport ', b'\nfrom ', b'\ndef ', b'\nclass ', b'\n# ', b'#!/']:
        idx = raw.find(p)
        if idx >= 0:
            return idx
    return 0

def clean_bytes(raw):
    cleaned = raw.replace(b'\x00', b'')
    start = find_code_start(cleaned)
    if start > 50:
        cleaned = cleaned[start:]
    text = cleaned.decode('utf-8', errors='replace')
    lines = []
    for line in text.split('\n'):
        if re.match(r'^w*s*l*:', line) or re.match(r'^N/e', line) or not line:
            continue
        lines.append(line)
    return '\n'.join(lines)

fixed = 0
for root, dirs, files in os.walk(BASE):
    for fn in files:
        if not fn.endswith('.py'):
            continue
        fp = os.path.join(root, fn)
        with open(fp, 'rb') as f:
            raw = f.read()
        if b'wsl: ' not in raw[:50] and b'N/e' not in raw[:20] and b'\x00' not in raw:
            continue
        clean_text = clean_bytes(raw)
        tmp = f'/tmp/_clean_{fn}'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        os.system(f'cp {tmp} "{fp}"')
        print(f"  Fixed: {fn} -> {len(raw)}b")
        fixed += 1

print(f"\nFixed {fixed} files")

# Verify
print("\nFiles needing attention:")
for root, dirs, files in os.walk(BASE):
    for fn in sorted(files):
        if not fn.endswith('.py'):
            continue
        fp = os.path.join(root, fn)
        with open(fp, 'rb') as f:
            start = f.read(30)
        if b'wsl:' in start or b'\x00' in start:
            print(f"  STILL BAD: {fn}")