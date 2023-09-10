# -*- coding: utf-8 -*-
from pprint import pprint
from utils.util import get_common_params, dict_merge

def run(params):
    pprint(params)
    model = params['model']
    # for route prediction task
    if model == 'speed':
        from algorithm.speed.speed import main
        main(params)
    if model == 'lgb':
        from algorithm.lgb.train import main
        main(params)
    if model == 'knn':
        from algorithm.lgb.train import main
        main(params)
    if model == 'mlp':
        from algorithm.mlp.train import main
        main(params)
    if model == 'ranketpa':
        from algorithm.rankepta.train import main
        main(params)
    if model == 'fdnet':
        from algorithm.fdnet.train import main
        main(params)

def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    params = vars(get_params())
    params['cuda_id'] = 0
    datasets = ['delivery_cq'] # the name of datasets
    args_lst = []
    params['is_test'] = False
    for model in ['fdnet']:
        if model in ['speed', 'lgb', 'knn']:
            for dataset in datasets:
                basic_params = dict_merge([params, {'model': model,'dataset': dataset}])
                args_lst.append(basic_params)

        if model in ['mlp', 'ranketpa', 'fdnet']:
            for hs in [32, 64]:
                for dataset in datasets:
                    deeproute_params = {'model': model, 'hidden_size': hs, 'dataset': dataset}
                    deeproute_params = dict_merge([params, deeproute_params])
                    args_lst.append(deeproute_params)



    # note: here you can use parallel running to accelerate the experiment.
    for p in args_lst:
        run(p)
