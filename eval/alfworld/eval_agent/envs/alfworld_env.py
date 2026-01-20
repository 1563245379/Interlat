import re
import json
import logging
from typing import Any, Dict, List, Tuple

from envs import BaseEnv
from tasks import AlfWorldTask
from prompt import prompt_with_icl
from utils.datatypes import State


logger = logging.getLogger("agent_frame")


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


class AlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.state = State()

    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    # def conduct_action(self, action: str):
    #     observation, reward, done, info = self.env.step([action])
    #     observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
    #     return observation, reward, done

    def conduct_action(self, action: str):
        print(f"Attempting action: '{action}'")
        observation, reward, done, info = self.env.step([action])
        
        # # 添加调试信息
        # print(f"Done: {done}, Reward: {reward}")
        # print(f"Info: {info}")
        # print(f"Raw observation: {observation}")
        
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        return observation, reward, done
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
            observation, reward, done = self.conduct_action(action)
        except Exception as e:
            # logger.debug(f"Agent failed with error: {e}")
            self.state.success = False
            self.state.finished = False
            self.state.reward=0
            observation = f"Observation: Error Input. Your input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state


        observation = f"Observation: {observation}"
        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward

        return observation, self.state

    def reset(self, game_files=None) -> Tuple[str, State]:
        self.state = State()
        self.env.reset_states(game_files)
        self.state.error = self.task.game_file
        cur_task = self.task.observation
        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 0)  #######
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state

    # def reset(self, game_files=None) -> Tuple[str, State]:
    #     self.state = State()

    #     # 使用标准 reset 接口，尝试传入 game_files
    #     try:
    #         obs, info = self.env.reset(options={"game_files": game_files})
    #     except TypeError:
    #         # 如果 options 不被支持，尝试无参数 reset
    #         obs, info = self.env.reset()

    #     print(f"admissible_commands: {info['admissible_commands'][0]}")

    #     # 更新 observation（如果需要）
    #     self.state.error = self.task.game_file
    #     cur_task = self.task.observation
    #     observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 0)

    #     if self.icl_format == 'first':
    #         self.state.history.append({
    #             "role": "user",
    #             "content": observation,
    #         })
    #     elif self.icl_format == 'conversation':
    #         self.state.history = messages

    #     return observation, self.state


    # def reset(self, game_files=None) -> Tuple[str, State]:
    #     self.state = State()
        
    #     # # 打印调试信息
    #     # print(f"Reset called with game_files: {game_files}")
    #     # print(f"Task game_file: {getattr(self.task, 'game_file', 'Unknown')}")
    #     # print(f"Environment type: {type(self.env)}")
        
    #     # # 检查环境是否是 TextworldBatchGymEnv
    #     # env_type = type(self.env).__name__
    #     # print(f"Environment class: {env_type}")
        
    #     # # 在reset之前检查环境状态
    #     # if hasattr(self.env, 'obs'):
    #     #     print(f"Before reset - env.obs[0]: {self.env.obs[0] if self.env.obs else 'Empty obs'}")
        
    #     # try:
    #     #     if env_type == 'TextworldBatchGymEnv':
    #     #         # TextworldBatchGymEnv 的特殊处理
    #     #         print("Using TextworldBatchGymEnv reset method")
                
    #     #         # 方法1：尝试直接传入game_files到reset
    #     #         if game_files is not None:
    #     #             try:
    #     #                 obs, info = self.env.reset(game_files)
    #     #                 print("Reset with game_files as direct parameter successful")
    #     #             except Exception as e:
    #     #                 print(f"Direct game_files parameter failed: {e}")
    #     #                 # 方法2：先设置游戏文件再reset
    #     #                 try:
    #     #                     if hasattr(self.env, 'set_game_files'):
    #     #                         self.env.set_game_files(game_files)
    #     #                         obs, info = self.env.reset()
    #     #                         print("Reset with set_game_files successful")
    #     #                     elif hasattr(self.env, 'game_files'):
    #     #                         self.env.game_files = game_files
    #     #                         obs, info = self.env.reset()
    #     #                         print("Reset with direct game_files assignment successful")
    #     #                     else:
    #     #                         print("No way to set game files, using default reset")
    #     #                         obs, info = self.env.reset()
    #     #                 except Exception as e2:
    #     #                     print(f"Alternative methods failed: {e2}")
    #     #                     obs, info = self.env.reset()
    #     #         else:
    #     #             obs, info = self.env.reset()
    #     #             print("Reset without game_files")
    #     #     else:
    #     #         # 其他环境类型的处理
    #     #         if game_files is not None:
    #     #             try:
    #     #                 obs, info = self.env.reset(options={"game_files": game_files})
    #     #                 print("Standard reset with options successful")
    #     #             except Exception as e:
    #     #                 print(f"Standard reset failed: {e}")
    #     #                 obs, info = self.env.reset()
    #     #         else:
    #     #             obs, info = self.env.reset()
                    
    #     # except Exception as e:
    #     #     print(f"All reset methods failed: {e}")
    #     #     # 最后的fallback
    #     #     obs, info = self.env.reset()
        
    #     # # 在reset之后检查环境状态
    #     # if hasattr(self.env, 'obs'):
    #     #     print(f"After reset - env.obs[0]: {self.env.obs[0] if self.env.obs else 'Empty obs'}")
        
    #     # # 检查obs的格式
    #     # if isinstance(obs, list) and len(obs) > 0:
    #     #     actual_obs = obs[0]
    #     # elif isinstance(obs, str):
    #     #     actual_obs = obs
    #     # else:
    #     #     print(f"Unexpected obs format: {type(obs)}, {obs}")
    #     #     actual_obs = str(obs)
        
    #     # print(f"Game file: {getattr(self.task, 'game_file', 'Unknown')}")
    #     # print(f"Task observation: {self.task.observation}")
    #     # print(f"Actual environment observation: {actual_obs}")
        
    #     # # 检查任务匹配
    #     # task_content = self.task.observation
    #     # if task_content and task_content not in actual_obs:
    #     #     print(f"WARNING: Task mismatch detected!")
    #     #     print(f"Expected task: {task_content}")
    #     #     print(f"Environment task: {actual_obs}")
            
    #     #     # 使用任务对象的观察作为实际观察
    #     #     print("Using task observation to ensure consistency")
    #     #     actual_obs = task_content
        
    #     # # 检查 admissible commands
    #     # if isinstance(info, dict) and 'admissible_commands' in info:
    #     #     admissible = info['admissible_commands']
    #     #     if isinstance(admissible, list) and len(admissible) > 0:
    #     #         if isinstance(admissible[0], list):
    #     #             print(f"Admissible commands: {admissible[0]}...")  # 显示前5个
    #     #         else:
    #     #             print(f"Admissible commands: {admissible}...")
    #     # else:
    #     #     print(f"Info structure: {info}")
        
    #     # # 使用实际环境的观察
    #     # observation, messages = prompt_with_icl(self.instruction, self.raw_icl, actual_obs, 0)

    #     # # messages[2]['content'] = messages[2]['content'][0][37:].replace('\n\n', '\n')
        
    #     # if self.icl_format == 'first':
    #     #     self.state.history.append({
    #     #         "role": "user",
    #     #         "content": observation,
    #     #     })
    #     # elif self.icl_format == 'conversation':
    #     #     self.state.history = messages

    #     obs, infos = self.env.reset_states(game_files)
            
    #     return observation, self.state

class BatchAlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        batch_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.batch_size = batch_size
        self.state = [State() for i in range(batch_size)]

    def parse_action(self, llm_output: List[str]) -> List[str]:
        llm_output = [x.strip() for x in llm_output]
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = [re.findall(pattern, x)[0] for x in llm_output]
        # assert action is not None
        return action
    
    def conduct_action(self, actions: List[str]):
        observation, reward, done, info = self.env.step(actions)
        outputs = []
        for i in range(self.batch_size):
            observation, reward, done = process_ob(observation[i]), info['won'][i], done[i]
            outputs.append((observation, reward, done))
        return outputs

    # def conduct_action(self, action: str):
    #     print(f"Attempting action: '{action}'")
    #     observation, reward, done, info = self.env.step([action])
        
    #     # 添加调试信息
    #     print(f"Done: {done}, Reward: {reward}")
    #     print(f"Info: {info}")
    #     print(f"Raw observation: {observation}")
        
    #     observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
    #     return observation, reward, done
    
    def step(self, llm_output: List[str]) -> Tuple[str, State]:
        for i in range(self.batch_size):
            self.state[i].history.append({
                "role": "assistant",
                "content": llm_output[i]
            })
        actions = self.parse_action(llm_output)
        
        observations = {}
        correct_idx = []
        
        for i, action in enumerate(actions):
            if action is None:
                self.state[i].success = False
                self.state[i].finished = False
                self.state[i].reward=0
                observation = f"Observation: Error Input. Your input must contains 'Action: '"
                self.state[i].history.append({
                    "role": "user",
                    "content": observation,
                })
                self.state[i].steps += 1
                if self.state[i].steps >= self.max_steps:
                    self.state[i].finished = True
                    self.state[i].success = False
                    self.state[i].terminate_reason = "max_steps"
                    self.state[i].reward = 0
                actions[i] = ""
                observations[i] = observation
            else:
                correct_idx.append(i)
        outputs = self.conduct_action(actions)
        for i in correct_idx:
            observation, reward, done = outputs[i]
            observation = f"Observation: {observation}"
            self.state[i].history.append({
                "role": "user",
                "content": observation,
            })

            self.state[i].steps += 1
            if self.state[i].steps >= self.max_steps:
                self.state[i].finished = True
                self.state[i].success = False
                self.state[i].terminate_reason = "max_steps"
                self.state[i].reward = reward

            if done:
                self.state[i].finished = True
                self.state[i].success = True
                self.state[i].terminate_reason = "success"
                self.state[i].reward = reward
            observations[i] = observation

        return list(observations.values), self.state

    def reset(self, game_files=None) -> Tuple[str, State]:
        self.state = [State() for i in range(self.batch_size)]
        # self.env.reset_states(game_files)
        cur_task = self.task.observation
        for i in range(self.batch_size):
            self.state[i].error = self.task.game_file
            obs = self.env.obs[i]
            obs = "\n".join(obs.split("\n\n")[1:])
            observation, messages = prompt_with_icl(self.instruction, self.raw_icl, obs, 0)
            if self.icl_format == 'first':
                self.state[i].history.append({
                    "role": "user",
                    "content": observation,
                })
            elif self.icl_format == 'conversation':
                self.state[i].history = messages
        return observation, self.state