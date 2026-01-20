import os
import json
import yaml
import logging
from typing import Iterable, Tuple
import sys

import alfworld
import alfworld.agents.environment as envs
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment import get_environment

from tasks.base import Task


logger = logging.getLogger("agent_frame")

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """Alfworld task instance."""

    task_name = "alfworld"

    def __init__(
        self,
        game_file: str,
        # env: envs.AlfredTWEnv,
        env,
        obs: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_file = game_file
        self.observation = obs

        self.env = env

    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1, batch_size: int = 1) -> Tuple[Iterable[Task], int]:
        # Method 1: Modify configuration data path to switch splits
        def set_split(config, split_name):
            # Use environment variable or fallback to default cache location
            alfworld_cache = os.environ.get('ALFWORLD_CACHE_PATH', os.path.expanduser('~/.cache/alfworld'))

            if split_name == 'train':
                config['dataset']['data_path'] = os.path.join(alfworld_cache, 'json_2.1.1/train')
            elif split_name == 'eval_in_distribution':
                config['dataset']['data_path'] = os.path.join(alfworld_cache, 'json_2.1.1/valid_seen')
            elif split_name == 'eval_out_of_distribution':
                config['dataset']['data_path'] = os.path.join(alfworld_cache, 'json_2.1.1/valid_unseen')
            return config

        # Set ALFWORLD_DATA environment variable if not already set
        if "ALFWORLD_DATA" not in os.environ:
            # Use relative path from project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
            default_alfworld_data = os.path.join(project_root, "eval", "alfworld", "eval_agent", "data", "alfworld")
            os.environ["ALFWORLD_DATA"] = default_alfworld_data
        alfworld_data_path = os.environ.get("ALFWORLD_DATA")

        try:
            with open(os.path.join(alfworld_data_path, "base_config.yaml")) as f:
                config = yaml.safe_load(f)
        except (FileNotFoundError, OSError):
            # 如果失败，设置环境变量并重试
            os.environ["ALFWORLD_DATA"] = "Interlat_preview/eval/alfworld/eval_agent/data/alfworld"
            fallback_path = os.environ["ALFWORLD_DATA"]
            with open(os.path.join(fallback_path, "base_config.yaml")) as f:
                config = yaml.safe_load(f)
        
        if split == 'train':
            split = "train"
            N_TASKS = 3321
        elif split == 'dev':
            split = "eval_in_distribution"
            N_TASKS = 140
        elif split == 'test':
            split = "eval_out_of_distribution"
            N_TASKS = 134

        env_type = config['env']['type']

        # 选择你想要的split
        desired_split = split  # 或 'train', 'valid_unseen'
        config = set_split(config, desired_split)

        env_type = config['env']['type']
        env = get_environment(env_type)(config, train_eval='train')  # 这里的train_eval参数可能不是用来指定split的
        env = env.init_env(batch_size=batch_size)

        if part_num > 1:
            assert part_idx != -1
            per_part_num = N_TASKS // part_num + 1
            skip_num = per_part_num * part_idx
            env.skip(skip_num)
            N_TASKS = min(per_part_num, N_TASKS - skip_num)
            split_index = range(skip_num, skip_num + N_TASKS)
        else:
            split_index = range(N_TASKS)

        def generator():
            for idx in split_index:
                obs, info = env.reset()
                obs = "\n".join(obs[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]

                yield cls(
                    task_id=idx,
                    game_file=game_file,
                    env=env,
                    obs=obs,
                )

        return generator(), N_TASKS
