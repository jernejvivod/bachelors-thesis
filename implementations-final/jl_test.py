from joblib import Parallel, delayed
from collections import namedtuple

resTuple = namedtuple('resTuple', 'name result')

results = set()
def collect_result(res):
    results.add(res)

def f1():
    print("tralala")

def f2():
    results.add('hopsasa')

def f3():
    return 'hop na klop'

functions = {
    'f1' : f1,
    'f2' : f2,
    'f3' : f3
}
