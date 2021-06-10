"""Test all supported algorithms

Not supported, Maze supported
            "A2C": _import_a2c,
            "A3C": _import_a3c,
            "APEX": _import_apex,
"APEX_DDPG": _import_apex_ddpg,
            "APPO": _import_appo,
"ARS": _import_ars,
"BC": _import_bc,
"ES": _import_es,
"DDPG": _import_ddpg,
            "DDPPO": _import_ddppo,
            "DQN": _import_dqn,
"DREAMER": _import_dreamer,
            "IMPALA": _import_impala,
"MAML": _import_maml,
            "MARWIL": _import_marwil,
"MBMPO": _import_mbmpo,
            "PG": _import_pg,
            "PPO": _import_ppo,
"QMIX": _import_qmix,
"SAC": _import_sac,
            "SimpleQ": _import_simple_q,
"TD3": _import_td3,

"""

import os
from typing import Dict

import pytest

from maze.test.shared_test_utils.hydra_helper_functions import run_maze_from_str

trainings = [
    {"rllib/algorithm": "a2c"},
    {"rllib/algorithm": "a3c"},
    {"rllib/algorithm": "appo"},
    {"rllib/algorithm": "ddppo"},
    {"rllib/algorithm": "impala"},
    {"rllib/algorithm": "ppo"},
    {"rllib/algorithm": "marwil"},
    {"rllib/algorithm": "simple_q"},
    {"rllib/algorithm": "dqn"},
    # {"rllib/algorithm": "apex"}, # testing apex without a GPU does not make much sense
]

# set configuration to test if not specified
training_defaults = {'rllib/runner': 'local', 'configuration': 'test',
                     'algorithm.config.num_gpus': '0', 'algorithm.config.num_gpus_per_worker': '0',
                     'runner.ray_config.ignore_reinit_error': 'True',
                     'env': 'gym_env', 'model': 'vector_obs', 'wrappers': 'vector_obs', 'critic': 'template_state'}


trainings = [pytest.param({**training_defaults, **t}, id="-".join(t.values())) for t in trainings]


@pytest.mark.rllib
@pytest.mark.parametrize("hydra_overrides", trainings)
def test_rllib_algorithms(hydra_overrides: Dict[str, str], tmpdir: str):
    # set working directory to temp path
    os.chdir(tmpdir)

    run_maze_from_str('maze.conf', 'conf_rllib', hydra_overrides)
