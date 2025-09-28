#!/usr/bin/env python3
"""Test retrieval quality"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.rag.retrieval import retrieve

# Test query
results = retrieve('What causes engine vibration during flight?', k=3)

print('=== RETRIEVAL TEST ===')
print(f"Found {len(results)} results\n")

for i, result in enumerate(results, 1):
    print(f'Result {i}: Score={result["score"]:.3f}')
    print(f'Source: {result["metadata"]["source"]}')
    print(f'Text: {result["text"][:200]}...')
    print('-' * 50)