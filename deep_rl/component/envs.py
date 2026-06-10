#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import multiprocessing as mp
import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from gymnasium.wrappers import AtariPreprocessing

from ..utils import *

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass


def is_atari_id(env_id):
    if env_id.startswith('ALE/') or 'NoFrameskip' in env_id:
        return True
    try:
        spec = gym.spec(env_id)
    except Exception:
        return False
    return 'ale_py' in str(spec.entry_point)


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
# a plain class instead of a closure so it can be pickled into subprocesses
class EnvMaker:
    def __init__(self, env_id, seed, rank, episode_life=True):
        self.env_id = env_id
        self.seed = seed
        self.rank = rank
        self.episode_life = episode_life

    def __call__(self):
        env_id = self.env_id
        random_seed(self.seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
            is_atari = False
        else:
            is_atari = is_atari_id(env_id)
            if is_atari:
                # AtariPreprocessing applies its own frame skip
                env = gym.make(env_id, frameskip=1)
            else:
                env = gym.make(env_id)
        # tracks the full-game return below the life-loss wrapper
        env = OriginalReturnWrapper(env, seed=self.seed + self.rank)
        if is_atari:
            env = AtariPreprocessing(env,
                                     frame_skip=4,
                                     terminal_on_life_loss=self.episode_life,
                                     grayscale_obs=True,
                                     scale_obs=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
        env = ClassicAPIWrapper(env)
        if is_atari:
            env = FrameStack(env, 4)

        return env


def make_env(env_id, seed, rank, episode_life=True):
    return EnvMaker(env_id, seed, rank, episode_life)


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env, seed=None):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0
        self._seed = seed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_rewards += reward
        if terminated or truncated:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self._seed is not None and 'seed' not in kwargs:
            kwargs['seed'] = self._seed
            self._seed = None
        return self.env.reset(**kwargs)


# collapses the gymnasium 5-tuple step API back to the classic
# (obs, reward, done, info) consumed by the agents
class ClassicAPIWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low.flat[0],
            self.observation_space.high.flat[0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.asarray(observation).transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


# operates on the classic 4-tuple API, above ClassicAPIWrapper
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = []
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 2:
            obs_shape = (1,) + obs_shape
        self.observation_space = Box(
            low=0, high=255,
            shape=(obs_shape[0] * k,) + obs_shape[1:],
            dtype=env.observation_space.dtype)

    def _to_frame(self, obs):
        obs = np.asarray(obs)
        if obs.ndim == 2:
            obs = obs[np.newaxis]
        return obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames = [self._to_frame(obs)] * self.k
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames = self.frames[1:] + [self._to_frame(obs)]
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class VecEnv:
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()


def _subproc_worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, rew, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, rew, done, info))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'close':
                remote.close()
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# adapted from baselines.common.vec_env.subproc_vec_env
class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.ps = [ctx.Process(target=_subproc_worker, args=(work_remote, remote, env_fn), daemon=True)
                   for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, n_envs, observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rew, done, info = zip(*results)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=None):
        if seed is None:
            seed = np.random.randint(int(1e9))
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = int(self.action_space.n)
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v4', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.env.num_envs, task.action_dim)
        next_state, reward, done, _ = task.step(action)
        print(done)
