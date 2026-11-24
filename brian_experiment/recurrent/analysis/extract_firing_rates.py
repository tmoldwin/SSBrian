import pickle
import os

folder = '../Runs/weight_scale_sweep_all_connections_E200_I200/20251115_180259'
types = ['E_to_E', 'E_to_I', 'I_to_E', 'I_to_I']

print('='*100)
print("FIRING RATE SUMMARY - ALL SWEEPS")
print('='*100)

for stype in types:
    print(f'\n{stype}:')
    print(f'{"Scale":<12} {"E Rate (Hz)":<15} {"I Rate (Hz)":<15} {"E/I Ratio":<12}')
    print('-'*60)
    
    with open(f'{folder}/sweep_results_{stype}.pkl', 'rb') as f:
        results = pickle.load(f)
    for r in results:
        if r.get('success'):
            scale = r.get('varied_scale', 0)
            e_rate = r.get('e_rate', 0)
            i_rate = r.get('i_rate', 0)
            ratio = e_rate / i_rate if i_rate > 0 else 0
            print(f'{scale:<12.4f} {e_rate:<15.2f} {i_rate:<15.2f} {ratio:<12.3f}')

print('\n' + '='*100)

