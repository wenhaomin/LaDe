# -*- coding: utf-8 -*-

from pprint import pprint
from utils.util import get_common_params, dict_merge
def run(params):
    pprint(params)
    model = params['model']
    if model == 'speed':
        from algorithm.speed.speed import main
        main(params)
    if model == 'lgb':
        from algorithm.lgb.train import main
        main(params)
    if model == 'knn':
        from algorithm.knn.train import main
        main(params)
    if model == 'mlp':
        from algorithm.mlp.train import main
        main(params)
    if model == 'ranketpa_route':
        from algorithm.ranketpa.train_route import main
        main(params)
    if model == 'ranketpa_time':
        from algorithm.ranketpa.train import main
        main(params)
    if model == 'fdnet':
        from algorithm.fdnet.train import main
        main(params)
    if model == 'm2g4rtp_delivery':
        from algorithm.m2g4rtp_delivery.train import main
        main(params)

def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    params = vars(get_params())
    datasets = ['delivery_yt', 'delivery_sh', 'delivery_cq'] # the name of datasets
    args_lst = []
    params['is_test'] = False
    params['inference'] = False
    for model in ['mlp' ]:
        if model in ['speed', 'lgb', 'knn']:
            for dataset in datasets:
                basic_params = dict_merge([params, {'model': model,'dataset': dataset}])
                args_lst.append(basic_params)

        if model in ['mlp', 'ranketpa_route', 'ranketpa_time', 'm2g4rtp_delivery', 'fdnet']:
            for hs in [64, 64, 64]:
                for dataset in datasets:
                    deeproute_params = {'model': model, 'hidden_size': hs, 'dataset': dataset}
                    deeproute_params = dict_merge([params, deeproute_params])
                    args_lst.append(deeproute_params)

    for p in args_lst:
        run(p)
