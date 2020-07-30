import os
import sys
import json
import networkx as nx
import pandas as pd
from tqdm import tqdm


def get_node_list():
    vertices = set()
    for label in ['malware', 'benign']:
        path = 'input/list_' + label + '.csv'
        file_list = pd.read_csv(path).values.flatten()
        root = '/media/ais/data/final_report_' + label + '/'

        for folder in tqdm(file_list, desc='Get nodes of ' + label):
            rp_path = root + folder
            for _, _, files in os.walk(rp_path):
                for file_name in files:
                    if file_name.startswith('strace'):
                        file_path = rp_path + '/' + file_name
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        for syscall in data:
                            vertices.add(syscall['name'])
    with open('temp/list_node.txt', 'w') as f:
        f.writelines('\n'.join(list(vertices)))


def scg():
    with open('temp/list_node.txt') as f:
        vertices = f.read().split('\n')

    for label in ['malware', 'benign']:
        path = 'input/list_' + label + '.csv'
        file_list = pd.read_csv(path).values.flatten()
        root = '/media/ais/data/final_report_' + label + '/'

        for folder in tqdm(file_list, desc='Generate SCGs of ' + label):
            rp_path = root + folder
            G = {'edges': list()}
            for _, _, files in os.walk(rp_path):
                for file_name in files:
                    if file_name.startswith('strace'):
                        file_path = rp_path + '/' + file_name
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        u = -1
                        for syscall in data:
                            if syscall['name'] != 'fork':
                                v = vertices.index(syscall['name'])
                                if u >= 0 and u != v and [u, v] not in G['edges']:
                                    G['edges'].append([u, v])
                                u = v
                            else:
                                u = -1
            with open('input/scg/' + folder + '.json', 'w') as f:
                json.dump(G, f)


if __name__ == "__main__":
    # get_node_list()
    scg()
