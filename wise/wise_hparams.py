from dataclasses import dataclass
from typing import List, Union
from ...util.hparams import HyperParams
import yaml


@dataclass
class WISEHyperParams(HyperParams):
    # Experiments

    edit_lr: float
    n_iter: int
    # Method
    objective_optimization: str
    mask_ratio: float
    alpha: float    # act_margin[0]
    beta: float  # act_margin[1]
    gamma: float  # act_margin[2]
    act_ratio: float
    merge_freq: int
    retrieve: bool
    replay: bool
    save_freq: Union[int, None]
    merge_alg: str
    norm_constraint: float
    # Module templates
    inner_params: List[str]
    weights: Union[float, None]
    densities: Union[float, None]

    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False

    # Save and Load
    save_path: str = None
    load_path: str = None

    grad_sort: bool = False
    merge_sort: bool = False
    grad_cluster: bool = False
    onebyone: bool = False
    maxsimseq: bool = False
    randomseq: bool = False
    subincluster: bool = False
    dynamic_cluster: bool = False

    cluster_sort: str = None
    my_batch_edit: str = None
    iter_batch_size: int = 5
    sensemask: bool = False
    sim_threshold: float = 0.5
    cfl_threshold: float = 0.2
    merge_moe: str = None

    merge_moe_freq: int = 10
    max_merge_num: int = 10
    init_train_ratio: float = 0.70
    act_loss_weight: float = 1.0
    lora_rank: int = 16
    adapter_scale: int = 32
    grad_samples_select: bool = True
    grad_sensetive_area: bool = False

    max_sample_num_per_cluster: int = 5
    activations_dtype: str = ''
    select_logic_1: bool = False
    select_logic_A800: bool = False
    select_all: bool = False
    no_A_in_produ: bool = False
    drop_regu: bool = False
    new_lora_dir: bool = False # A0 - A, B0 - B -> cluster['grad_matrix']
    regu_loss_weight: float = 1.0
    select_logic_new0507: bool = False
    bound_ratio: float = 0.5
    act_key_last: bool = False
    use_gamma_para: bool = False
    ga_para_0: float = 0.6
    ga_para_1: float = 2.0
    rand_mask: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert config['merge_freq'] % config['save_freq'] == 0, 'merge_freq need to be divisible by save_freq (like 1000 / 500)'
        assert len(config['act_margin']) == 3
        config['alpha'], config['beta'], config['gamma'] = config['act_margin'][0], config['act_margin'][1], config['act_margin'][2]
        config.pop('act_margin')

        assert (config and (config['alg_name'] == 'WISE' or config['alg_name'] == 'GRADMOE' or config['alg_name'] == 'CAN')), \
            f'WISEHyperParams can not load from {hparams_name_or_path}. alg_name is {config["alg_name"]}'
        return cls(**config)