import pandas as pd


psi = pd.read_csv('output/psig.csv')
scg = pd.read_csv('output/scg.csv')
fusion = list()
for row in psi.values:
    name = row[0]
    scg_row = scg.loc[scg['name']==name, 'x_0':].values.flatten()
    fusion_row = list(row[:-1])
    fusion_row.extend(scg_row)
    fusion.append(fusion_row)
header = ['name'] + [f'x_{i}' for i in range(256)] + ['label']
pd.DataFrame(fusion, columns=header).to_csv('output/fusion.csv', index=None)
