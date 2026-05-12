from src.lab24_common import cohen_kappa
import csv

judge=[]
human=[]
with open('phase-b/pairwise_results.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        judge.append(row['winner_after_swap'])
with open('phase-b/human_labels.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        human.append(row['human_winner'])
print(round(cohen_kappa(judge[:len(human)], human), 4))
