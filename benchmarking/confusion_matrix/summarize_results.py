from pathlib import Path
from collections import defaultdict
import json


alo = defaultdict(list)
cv = defaultdict(list)
test = defaultdict(list)

def load(directory):
    for path in Path(directory).iterdir():
        with path.open() as o:
            result = json.load(o)
        lamda = result['config']['method_kwargs']['lamda0']
        seed = result['config']['seed']
        alo[seed].append((lamda, result['alo_100_poly_risk']))
        cv[seed].append((lamda, result['cv_5_risk']))
        test[seed].append((lamda, result['test_risk']))
load('results/')

def summarize(d):

    summary = defaultdict(lambda : 0)
    for v in d.values():
        if len(v) > 2:
            print(v)
        lamda_max = None
        risk_max = float('inf')
        for lamda, r1 in  v:
            if r1 < risk_max:
                lamda_max = lamda
                risk_max = r1
        summary[lamda_max] += 1
    return summary

print('ALO', summarize(alo))
print('CV', summarize(cv))
print('Test', summarize(test))

