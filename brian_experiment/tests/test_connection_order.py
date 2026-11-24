"""Test to verify Brian2's connection order"""
from brian2 import *
import numpy as np

prefs.codegen.target = 'numpy'

N_source = 3
N_target = 4

# Create neuron groups
source = NeuronGroup(N_source, 'v : 1')
target = NeuronGroup(N_target, 'v : 1')

# Create synapses
S = Synapses(source, target, 'w : 1')
S.connect()  # All-to-all

# Print connection order
print("Connection order (source -> target):")
for i in range(len(S)):
    print(f"  Connection {i}: source {S.i[i]} -> target {S.j[i]}")

# Create a test weight matrix [target, source]
weight_matrix = np.arange(N_target * N_source).reshape(N_target, N_source)
print(f"\nWeight matrix shape: {weight_matrix.shape} (target, source)")
print("Weight matrix:")
print(weight_matrix)

# Test extraction methods
print("\nMethod 1: weight_matrix[:, :N_source].flatten() (columns, all rows)")
extract1 = weight_matrix[:, :N_source].flatten()
print(f"  Extracted: {extract1}")
print(f"  Order: target-by-target, then source-by-source within each target")

print("\nMethod 2: weight_matrix[:N_source, :].T.flatten() (rows, transposed)")
extract2 = weight_matrix[:N_source, :].T.flatten()
print(f"  Extracted: {extract2}")
print(f"  Order: source-by-source, then target-by-target within each source")

print("\nExpected Brian order (source-major):")
expected = []
for s in range(N_source):
    for t in range(N_target):
        expected.append(weight_matrix[t, s])
print(f"  Expected: {expected}")

print(f"\nMatch with method 1: {np.allclose(extract1, expected)}")
print(f"Match with method 2: {np.allclose(extract2, expected)}")

