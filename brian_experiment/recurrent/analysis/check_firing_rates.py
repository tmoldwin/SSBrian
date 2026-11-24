import pickle
import os

folder = '../Runs/weight_scale_sweep_all_connections_E200_I200/20251115_164538'
types = ['E_to_E', 'E_to_I', 'I_to_E', 'I_to_I']

print('Previous firing rates:')
for stype in types:
    print(f'\n{stype}:')
    with open(f'{folder}/sweep_results_{stype}.pkl', 'rb') as f:
        results = pickle.load(f)
    for r in results:
        print(f'  Scale {r["varied_scale"]:.2f}: E={r["e_rate"]:.1f}Hz, I={r["i_rate"]:.1f}Hz')

