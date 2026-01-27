#!/usr/bin/env python3
"""
Quick test to verify the mutation testing script can actually run tests.
This will test just ONE mutation to verify everything works.
"""
import sys
sys.path.insert(0, 'mutation_testing_linear_algebra')
from single_file_mutation import MutationGenerator, MutationTester
from pathlib import Path

print("=" * 70)
print("QUICK VERIFICATION TEST")
print("=" * 70)

# Generate just one mutation
print("\n1. Generating one mutation...")
gen = MutationGenerator(Path('.'))
all_muts = gen.generate_all('optax/_src/linear_algebra.py', max_mutations=150)
if not all_muts:
    print("ERROR: No mutations generated!")
    sys.exit(1)

# Take the first mutation
test_mutation = all_muts[0]
print(f"   Selected mutation #{test_mutation.id} ({test_mutation.operator})")
print(f"   Line {test_mutation.line_number}: {test_mutation.original_code.strip()[:60]}...")

# Try to run the test
print("\n2. Testing if pytest can run...")
tester = MutationTester(Path('.'))
test_file = 'optax/_src/linear_algebra_test.py'

try:
    # First verify the test file runs without mutation
    print(f"   Running: {test_file}")
    passed, output = tester.run_targeted_test(test_file, timeout=30)
    
    if "ERROR" in output or "error during collection" in output:
        print("\n❌ PROBLEM: Test file has import/collection errors")
        print("Output:", output[:500])
        sys.exit(1)
    
    if passed:
        print("   ✅ Tests can run (original code passes)")
    else:
        print("   ⚠️  Original tests are failing (might be expected)")
    
    # Now test one mutation
    print(f"\n3. Applying mutation #{test_mutation.id}...")
    status = tester.test_mutation(test_mutation, test_file)
    
    print(f"   Result: {status}")
    
    if status == "killed":
        print("   ✅ Mutation was KILLED (test detected the change)")
    elif status == "survived":
        print("   ⚠️  Mutation SURVIVED (test didn't detect the change)")
    else:
        print(f"   ❌ ERROR: {status}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nThe script CAN execute tests. Ready for full mutation testing.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
