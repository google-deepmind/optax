#!/usr/bin/env python3
"""
Quick test to verify mutation testing now works with the fixed setup.
Tests just 3 mutations to validate the environment and script.
"""
import sys
sys.path.insert(0, 'mutation_testing_linear_algebra')
from single_file_mutation import MutationGenerator, MutationTester
from pathlib import Path

print("=" * 70)
print("QUICK MUTATION TEST (3 mutations)")
print("=" * 70)

# Generate mutations
print("\n1. Generating mutations...")
gen = MutationGenerator(Path('.'))
all_muts = gen.generate_all('optax/_src/linear_algebra.py', max_mutations=150)
print(f"   Generated {len(all_muts)} total mutations")

# Test first 3 mutations
test_mutations = all_muts[:3]
print(f"\n2. Testing first 3 mutations...")

tester = MutationTester(Path('.'))
test_file = 'optax/_src/linear_algebra_test.py'

results = {"killed": 0, "survived": 0, "errors": 0}

for i, mutation in enumerate(test_mutations, 1):
    print(f"\n   [{i}/3] Mutation #{mutation.id} ({mutation.operator}) - Line {mutation.line_number}")
    print(f"   Original: {mutation.original_code.strip()[:60]}...")
    
    try:
        status = tester.test_mutation(mutation, test_file)
        results[status] += 1
        
        if status == "killed":
            print(f"   ✅ KILLED - Test detected the mutation")
        elif status == "survived":
            print(f"   ⚠️  SURVIVED - Test did not detect the mutation")
        else:
            print(f"   ❌ ERROR: {status}")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        results["errors"] += 1

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")
print(f"Results: {results['killed']} killed, {results['survived']} survived, {results['errors']} errors")

if results["errors"] > 0:
    print("\n⚠️  There were errors. Check the output above.")
    sys.exit(1)
elif results["killed"] == 0 and results["survived"] == 3:
    print("\n⚠️  WARNING: All mutations survived (might indicate tests aren't detecting changes)")
    sys.exit(1)
elif results["killed"] == 3 and results["survived"] == 0:
    print("\n⚠️  WARNING: All mutations killed (check this isn't the old false positive)")
    # This is actually OK if tests are working, just noting it
else:
    print("\n✅ Mix of killed and survived - mutation testing is working correctly!")

print("\n🎉 Environment is set up correctly. Ready for full mutation testing!")
