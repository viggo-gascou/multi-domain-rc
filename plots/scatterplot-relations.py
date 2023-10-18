import json
import os
import matplotlib as mpl
import matplotlib.patches as mpatches
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

import torch
from sklearn.decomposition import PCA

import csv
data = []
all_embeddings = []
with open("embeddings_relations.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=';')
    for row in csvreader:
        if row[0] =='embeddings':
            continue
        representation = [float(x) for x in row[0].replace('[', '').replace(']', '').split(', ')]
        all_embeddings.append(representation)
        data.append(row[1:])

print("PCAing")
pca = PCA(n_components=2)
transformed = pca.fit_transform(all_embeddings)

fig, ax = plt.subplots(figsize=(15,12), dpi=300)

domain2marker = {}
markers = ['.', 'x', '4', '3', '_']
markers = markers + markers + markers + markers
domains = sorted(list(set([x[1] for x in data])))
for domainIdx, domain in enumerate(domains):
    domain2marker[domain] = markers[domainIdx]

rel2color = {}
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = colors + colors + colors
relation_labels = sorted(list(set([x[0] for x in data])))
for relIdx, rel in enumerate(relation_labels):
    rel2color[rel] = colors[relIdx]
print(len(relation_labels))

freqs = {}
for item in data:
    label = item[0]
    if label in freqs:
        freqs[label] += 1
    else:
        freqs[label] = 1

topN = [x[0] for x in sorted(freqs.items(), key=lambda x:x[1])[-8:]]

domainsAdded = set()
relsAdded = set()
for instanceIdx in range(len(all_embeddings)):
    coordinate = transformed[instanceIdx]
    relation, domain = data[instanceIdx]
    if  relation not in topN:
        continue
    color = rel2color[relation]
    marker = domain2marker[domain]
    if relation not in relsAdded:
        ax.scatter(coordinate[0], coordinate[1], color=color, marker=marker, label=relation, alpha=.7)
        relsAdded.add(relation)
    else:
        if relation == relation_labels[0]:
            if domain not in domainsAdded:
                ax.scatter(coordinate[0], coordinate[1], color=color, marker=marker, label=domain, alpha=.7)
                domainsAdded.add(domain)
            else:
                ax.scatter(coordinate[0], coordinate[1], color=color, marker=marker, alpha=.7)
        else:
            ax.scatter(coordinate[0], coordinate[1], color=color, marker=marker, alpha=.7)
    #if domain not in domainsAdded:
    #    domainsAdded.add(domain)
    #else:
    #    ax.scatter(coordinate[0], coordinate[1], color=domain2color[domain], alpha=.7)
    
leg = ax.legend()
leg.get_frame().set_linewidth(1.5)
fig.savefig('scatter_plot_relations.pdf', bbox_inches='tight')

