import argparse
from human_eval.data import write_jsonl, read_problems
import math

with open('log.txt') as file:
    text = file.read()
    parts = text.split('----------------------------------------------------------------\n')

parts = parts[:500]

samples = []
for i, sample in enumerate(parts):
    task_id = "HumanEval/{}".format(math.floor(i / 10))
    result = dict(
                    task_id=task_id,
                    completion=sample,
                )
    
    samples += [result]

write_jsonl('results/first_50_gamma_04_delta_10/eval.jsonl', samples)