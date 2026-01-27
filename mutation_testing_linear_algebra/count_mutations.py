#!/usr/bin/env python3
import sys
sys.path.insert(0, 'mutation_testing_linear_algebra')
from single_file_mutation import MutationGenerator
from pathlib import Path
from collections import Counter

gen = MutationGenerator(Path('.'))
muts = gen.generate_all('optax/_src/linear_algebra.py', max_mutations=150)
print(f'Total mutations: {len(muts)}')
ops = Counter(m.operator for m in muts)
for op, count in ops.items():
    print(f'  {op}: {count}')

print('\nFirst 10 mutations to verify they are in actual code:')
for i in range(min(10, len(muts))):
    m = muts[i]
    print(f'\nMutation {m.id} ({m.operator}) - Line {m.line_number}:')
    orig = m.original_code.strip()
    mut = m.mutated_code.strip()
    if len(orig) > 80:
        orig = orig[:77] + '...'
    if len(mut) > 80:
        mut = mut[:77] + '...'
    print(f'  Original: {orig}')
    print(f'  Mutated:  {mut}')

