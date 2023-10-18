import json
import os
import matplotlib as mpl
import matplotlib.patches as mpatches
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.decomposition import PCA

data_dir = 'crossre_data/'

all_embeddings = []
domain_labels = []

lm = 'bert-base-cased'
lang_model = AutoModel.from_pretrained(lm)
tokenizer = AutoTokenizer.from_pretrained(lm)

for annotated_file in os.listdir(data_dir):
    if 'train' in  annotated_file:
        print('reading and encoding: ' + annotated_file)
        domain = annotated_file.split('-')[0]
        for line in open(data_dir + annotated_file):#.readlines()[:10]:
            data = json.loads(line)
            text = ' '.join(data['sentence'])
            word_ids = tokenizer.encode(text, return_tensors='pt')
            embeds = lang_model.forward(word_ids)[0]
            # shape = 1, sent_len, 768
            # we need to average over middle dimension (words)
            avg_pooled = embeds.mean(dim=1).squeeze()
            all_embeddings.append(avg_pooled.detach().numpy())
            domain_labels.append(domain)

print("PCAing")
pca = PCA(n_components=2)
transformed = pca.fit_transform(all_embeddings)


fig, ax = plt.subplots(figsize=(8,5), dpi=300)
domains = sorted(list(set(domain_labels)))
domain2color = {}
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for domainIdx, domain in enumerate(domains):
    domain2color[domain] = colors[domainIdx]


# adds 1 by 1, perhaps not neatest?
domainsAdded = set()
for instanceIdx in range(len(all_embeddings)):
    coordinate = transformed[instanceIdx]
    domain = domain_labels[instanceIdx]
    if domain not in domainsAdded:
        ax.scatter(coordinate[0], coordinate[1], color=domain2color[domain], label=domain, alpha=.7)
        domainsAdded.add(domain)
    else:
        ax.scatter(coordinate[0], coordinate[1], color=domain2color[domain], alpha=.7)
    
leg = ax.legend()
leg.get_frame().set_linewidth(1.5)
fig.savefig('scatter_plot_domains.pdf', bbox_inches='tight')

