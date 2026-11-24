"""Test if the transpose bug violates Dale's Law"""
import numpy as np

# Simulate the weight matrix and extraction
N_REC_E = 3
N_REC_I = 2
N_rec = N_REC_E + N_REC_I

# Create weight matrix [target, source] with Dale's Law applied
weight_rec = np.random.randn(N_rec, N_rec)
weight_rec[:, :N_REC_E] = np.abs(weight_rec[:, :N_REC_E])  # E sources: positive
weight_rec[:, N_REC_E:] = -np.abs(weight_rec[:, N_REC_E:])  # I sources: negative

print("Weight matrix shape:", weight_rec.shape, "(target, source)")
print("\nWeight matrix (Dale's Law applied):")
print("  E sources (cols 0-2): all positive")
print("  I sources (cols 3-4): all negative")
print(weight_rec)

# Simulate Brian's connection order (source-major)
# For E sources: source 0 -> targets [0,1,2,3,4], source 1 -> targets [0,1,2,3,4], etc.
print("\n" + "="*60)
print("CORRECT extraction (with transpose):")
print("="*60)
weight_E_correct = weight_rec[:, :N_REC_E].T.flatten()
print(f"E weights extracted (correct order): {weight_E_correct}")
print(f"All positive? {np.all(weight_E_correct > 0)}")

# Simulate WRONG extraction (without transpose - what the bug does)
print("\n" + "="*60)
print("WRONG extraction (without transpose - the bug):")
print("="*60)
weight_E_wrong = weight_rec[:, :N_REC_E].flatten()
print(f"E weights extracted (wrong order): {weight_E_wrong}")
print(f"All positive? {np.all(weight_E_wrong > 0)}")

# Check if the wrong extraction still respects Dale's Law (all positive from E sources)
print("\n" + "="*60)
print("Dale's Law check:")
print("="*60)
print(f"Correct extraction - all E source weights positive: {np.all(weight_E_correct > 0)}")
print(f"Wrong extraction - all E source weights positive: {np.all(weight_E_wrong > 0)}")
print("\nConclusion: The wrong extraction still respects Dale's Law (all E source weights are positive)")
print("BUT: The weights are assigned to the WRONG connections!")

# Show what connections get what weights
print("\n" + "="*60)
print("Connection assignment comparison:")
print("="*60)
print("Brian connects in order: source0->target0, source0->target1, ..., source0->target4, source1->target0, ...")
print("\nCorrect assignment:")
for i, w in enumerate(weight_E_correct[:10]):
    source = i // N_rec
    target = i % N_rec
    print(f"  Connection {i}: E source {source} -> target {target}, weight = {w:.3f}")

print("\nWrong assignment (what actually happens with bug):")
for i, w in enumerate(weight_E_wrong[:10]):
    source = i // N_rec
    target = i % N_rec
    print(f"  Connection {i}: E source {source} -> target {target}, weight = {w:.3f}")

print("\nThe weights are shuffled but still all positive from E sources!")

