# %%
from NPSmethods import *
from NPSmethods import __calcNPS
methods = set()
connections = []
with open('NPSmethods.py', 'r') as f:
    for line in f:
        if line.startswith('def'):
            ind = line.index('(')
            meth = line[4:ind]
            methods.add(meth)
for m in methods:
    names = eval(f'{m}.__code__.co_names')
    for tm in set(names).intersection(methods):
        connections.append(f'{tm} -> {m};')
# print(connections)
# %%
for conn in connections:
    print(conn)
# %%
