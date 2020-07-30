import os
import sys
import json
import networkx as nx
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


def get_node_list():
    vertices = set()
    mal_list = pd.read_csv('input/list_malware.csv').values.flatten()
    beg_list = pd.read_csv('input/list_benign.csv').values.flatten()
    mal = [name[:name.find('_')] for name in mal_list]
    beg = [name[:name.find('_')] for name in beg_list]

    for folder in ['bashlite/', 'mirai/', 'others/', 'benign/']:
        path = 'input/psi_graph/' + folder
        for _, _, files in os.walk(path):
            for file in tqdm(files, desc=folder[:-1]):
                if file.replace('.txt', '') in mal or file.replace('.txt', '') in beg:
                    fpath = path + file
                    with open(fpath, 'r') as f:
                        data = f.read().split('\n')
                    for line in data[2:-1]:
                        e = line.split()
                        if len(e) == 2:
                            vertices.add(e[0])
                            vertices.add(e[1])
    with open('temp/list_node_psi.txt', 'w') as f:
        f.writelines('\n'.join(list(vertices)))


def psig():
    mal_list = pd.read_csv('input/list_malware.csv').values.flatten()
    beg_list = pd.read_csv('input/list_benign.csv').values.flatten()
    mal = [name[:name.find('_')] for name in mal_list]
    beg = [name[:name.find('_')] for name in beg_list]
    with open('temp/list_node_psi.txt') as f:
        vertices = f.read().split('\n')

    def run(folder):
        path = 'input/psi_graph/' + folder
        for _, _, files in os.walk(path):
            for file in tqdm(files, desc=folder[:-1]):
                if file.replace('.txt', '') in mal or file.replace('.txt', '') in beg:
                    G = {'edges': list()}
                    fpath = path + file
                    with open(fpath, 'r') as f:
                        data = f.read().split('\n')
                    for line in data[2:-1]:
                        e = line.split()
                        if len(e) == 2:
                            G['edges'].append(
                                [vertices.index(e[0]), vertices.index(e[1])])
                    with open('input/psig/' + file.replace('.txt', '.json'), 'w') as f:
                        json.dump(G, f)
    
    num_cores = multiprocessing.cpu_count()
    folders = ['bashlite/', 'mirai/', 'others/', 'benign/']
    results = Parallel(n_jobs=num_cores)(delayed(run)(i) for i in folders)


if __name__ == "__main__":
    # get_node_list()
    psig()
