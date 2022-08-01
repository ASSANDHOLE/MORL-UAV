import sys
from multiprocessing import set_start_method
import pickle

import numpy as np

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination

from all import multiprocessing_one_generation
from gym_uav_my import gen_obs_map
from gym_uav_my import NamedDict

config = NamedDict({
    'num_proc': 32, 'time_step': 200000, 'eval_time': 200,
    'available_devices': 'all', 'pop_size': 64  # offspring_size=pop_size
})


class MyProblem(Problem):
    def __init__(self, save_eval=False, dump_fn=None):
        lower_bound = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        upper_bound = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        self.eval_obs_map = [gen_obs_map(4, 0.5) for _ in range(config['eval_time'])]
        self.save_eval = save_eval
        self.eval_res = []
        # constraint: no collide with obstacle or wall, success rate >= ?%
        problem_kwargs = {
            'n_var': 5, 'n_obj': 2, 'n_constr': 2,
            'xl': lower_bound, 'xu': upper_bound
        }
        assert not (not save_eval and dump_fn is not None), '"dump_fn" is not None while save_eval is false'
        self.dump_fn = dump_fn
        super().__init__(**problem_kwargs)

    def run(self, x):
        info_list = multiprocessing_one_generation(
            config.num_proc, x, config.time_step,
            self.eval_obs_map, config.available_devices
        )
        res = []
        for x in info_list:
            suc = 0
            cli = 0
            tlr = 0
            path_ratio = []
            speed_ratio = []
            for y in x[1]:
                if y.done.value == 1:
                    suc += 1
                    path_ratio.append(y.length_shortest_ratio)
                    speed_ratio.append(y.speed_maximun_ratio)
                elif y.done.value == 2 or y.done.value == 3:
                    cli += 1
                else:
                    tlr += 1
            r_l = [y.risk_factor for y in x[1]]
            r_max = max(r_l)
            if len(path_ratio) > 0:
                path_ratio_mean = sum(path_ratio) / len(path_ratio)
            else:
                path_ratio_mean = 1000000
            if len(speed_ratio) > 0:
                speed_ratio_mean = sum(speed_ratio) / len(speed_ratio)
            else:
                speed_ratio_mean = 1000000
            # efficiency (smaller is better)
            # TODO: Is it better to use path and time, instead of speed
            obj1 = 0.5*path_ratio_mean + 0.5*speed_ratio_mean
            # safety (smaller is better)
            obj2 = r_max
            # no collision
            con1 = cli - 0.1
            # success rate >= 70%
            con2 = 0.69 - suc / len(x[1])
            res.append([obj1, obj2, con1, con2])
        return np.array(res)

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.run(x)
        f = np.column_stack([res[:, 0], res[:, 1]])
        g = np.column_stack([res[:, 2], res[:, 3]])
        if self.save_eval:
            self.eval_res.append((x, res, f, g))
            if self.dump_fn is not None:
                self.dump_fn(self.eval_res)
        out['F'] = f
        out['G'] = g


def run():
    def dump_fn(x):
        try:
            with open('./eval_his.pkl', 'wb') as pkl:
                pickle.dump(x, pkl)
        except:
            print('dump eval_history failed')

    problem = MyProblem(save_eval=True, dump_fn=dump_fn)
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
