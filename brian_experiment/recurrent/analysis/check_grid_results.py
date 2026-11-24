import pickle
import os
import glob

folders = glob.glob('../Runs/grid_search_EtoI_ItoE_E200_I200/*/grid_search_results.pkl')
folders.sort()
latest = folders[-1] if folders else None

if latest:
    print(f'Latest results: {latest}')
    with open(latest, 'rb') as f:
        results = pickle.load(f)
    print(f'Total results: {len(results)}')
    successful = [r for r in results if r.get('success')]
    print(f'Successful: {len(successful)}')
    print(f'Failed: {len(results) - len(successful)}')
    
    if successful:
        print(f'\nFirst successful result:')
        r = successful[0]
        print(f'  E-to-I scale: {r.get("e_to_i_scale")}')
        print(f'  I-to-E scale: {r.get("i_to_e_scale")}')
        print(f'  E rate: {r.get("e_rate")} Hz')
        print(f'  I rate: {r.get("i_rate")} Hz')
        print(f'  Has spike data: {r.get("spike_times_all") is not None}')
else:
    print('No results found')


