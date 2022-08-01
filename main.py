import sys
from multiprocessing import set_start_method
import pickle

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_crossover, get_mutation, get_termination

from gym_uav import NamedDict
from uav_problem import set_problem_config
from uav_problem import UavProblem

config = NamedDict({
    'num_proc': 32, 'time_step': 200000, 'eval_time': 200,
    'available_devices': 'all', 'pop_size': 64  # offspring_size=pop_size
})

set_problem_config(config)


def run():
    def dump_fn(x):
        try:
            with open('./eval_his.pkl', 'wb') as pkl:
                pickle.dump(x, pkl)
        except:
            print('dump eval_history failed')

    problem = UavProblem(save_eval=True, dump_fn=dump_fn)
    init_arr = np.array([[14.0, 14.0, 22.0, 22.0, 14.0], [7.0, 6.0, 5.0, 4.0, 3.0], [8.0, 1.0, 2.0, 5.0, 2.0], [25.0, 3.0, 2.0, 1.0, 1.0]])
    algorithm = NSGA2(
        pop_size=int(config.pop_size),
        n_offsprings=int(config.pop_size),
        # sampling=get_sampling('real_random'),
        sampling=init_arr,
        crossover=get_crossover('real_sbx', prob=0.1, eta=15),
        mutation=get_mutation('real_pm', prob=0.9, eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination('n_gen', 100)
    res = minimize(
        problem,
        algorithm,
        termination,
        save_history=True,
        verbose=True
    )
    try:
        with open('./res_mo.pkl', 'wb') as f:
            pickle.dump(res, f)
        print('dump res to ./res_mo.pkl')
    except:
        print('dump res failed')

    try:
        with open('./eval_his.pkl', 'wb') as f:
            pickle.dump(problem.eval_res, f)
        print('dump eval_history to ./eval_his.pkl')
    except:
        print('dump eval_history failed')

    return res, problem

 
if __name__ == '__main__':
    if sys.platform == 'linux':
        set_start_method('spawn')
    run()
