# stuff specifically for the sklearn logic
from typing import Mapping
from functools import partial, reduce
import operator

from itertools import product
import argparse

from human_eval.data import write_jsonl, read_problems
import json

###############################################################################
# A grid search convenience class
###############################################################################
class ParameterGrid:
    """logic YOINKED from sklearn <3
    def worth just using the lib itself, or something fancier in future for 
    efficient sampling etc. It's implemented as an iterator interface but thats
    probs not necessary"""

    def __init__(self, params):
        # we may want to product a few sets of parameters
        # independently of eachother, so expects a List[Mapping]
        if isinstance(params, Mapping):
            self.params = [params]
        else:
            self.params = params
        # removed all checking code soooo make sure your
        # param dict is already nice and conforming
    
    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.params:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params
    
    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(
            product(len(v) for v in p.values()) if p else 1 for p in self.params
        )

###############################################################################
# little "oneliner" reduce thingy that turns your shallow dict into
# the list [k1, v1, k2, v2, k3, v3 ...]
# and optionally "k1 v1 k2 v2 k3 v3"

def flatten_dict(dict, to_string=False, sep=" "):
    flat_dict = reduce(operator.iconcat,dict.items() , [])
    if to_string:
        try: 
            return sep.join([str(elm) for elm in flat_dict])
        except:
            raise ValueError(f'Error converting dict={flat_dict} to whitespace joined string')
    else:
        return flat_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_human_eval_groundtruth():
    problems = read_problems()

    result = [problems[task_id]['canonical_solution'] for task_id in problems]

    return result


def load_dataset_groundtruth(dataset_name):
    if dataset_name == 'human_eval':
        return load_human_eval_groundtruth()
    elif dataset_name == 'mbpp':
        raise Exception("Not implemented for mbpp yet")
    else:
        raise Exception("Unknown dataset name")
    

def load_generated_code(file_path):
    problems = read_problems()

    prompt = {task_id:problems[task_id]['prompt'] for task_id in problems}

    result = []
    with open(file_path + '/eval.jsonl', 'r') as file:
            for i, line in enumerate(file):
                json_data = json.loads(line)
                result.append(json_data['completion'][len(prompt[json_data['task_id']]):])
                
    return result