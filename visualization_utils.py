import matplotlib.pyplot as plt
from uav_problem import UavProblem
import pickle


def load_data(path):
    with open(path, 'rb') as f:
        # import UavProblem is required to load res_mo.pkl
        # if name not match, you can use "import UavProblem as ..."
        data = pickle.load(f)
    return data


def get_populations_info(res_mo):
    history = res_mo.history
    pops = [h.off for h in history]
    val = [[p.F for p in pop] for pop in pops]
    param = [[p.X for p in pop] for pop in pops]
    opt_val = [o.F for o in res_mo.opt]
    opt_param = [o.X for o in res_mo.opt]
    return val, param, opt_val, opt_param


def draw_gif():
    pass
