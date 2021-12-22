# %%

import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np

from . import data

# %%

DEFAULT_BARS_COUNT = 0
DEFAULT_COMMISSION_PERC = 0
1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(
        self,
        bars_count,
        commission_perc,
        reset_on_close,
        reward_on_close=True,
        volumes=True,
    ):
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):

        if self.volumes:
            return (4 * self.bars_count + 1 + 1,)
        else:
            return (3 * self.bars_count + 1 + 1,)

    def encode(self):
        """
        Convert current state into numpy array
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volumes[self._offset + bar_idx]
                shift += 1
            res[shift] = float(self.have_position)
            shift += 1

            if not self.have_position:
                res[shift] = 0.0
            else:
                res[shift] = (self._cur_close() - self.open_price) / self.open_price
            return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[
            self._offset
        ]  # not e h,l,c are relative value with open price
        rel_close = self._prices.close[self._offset]  # store in ration
        return open * (
            1.0 + rel_close
        )  # so here,we just adding rel close price + open price = actual close price

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of
        prices,
        :param: action
        :return: reward,done
        """
        reward = 0.0
        done = False
        close = self._cur_close()
        # buy [note here we are talking about only one position per agent]
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close  # now the price at buying
            reward -= self.commission_perc
        # close the position
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close  # if reset on close its done

            if self.reward_on_close:
                reward += (
                    100.0 * (close - self.open_price) / self.open_price
                )  # reward as pecentage change in cost of position

            self.have_position = False
            self.open_price = 0.0


class State1D(State):
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    # here our stock as 2d matrix--> for 1d convolution
    def encode(self):
        res = np.array(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count - 1
        res[0] = self._prices.high[self._offset - ofs : self._offset + 1]
        res[1] = self._prices.low[self._offset - ofs : self._offset + 1]
        res[2] = self._prices.close[self._offset - ofs : self._offset + 1]
        dst = 3
        if self.volumes:
            res[3] = self._prices.volumn[self._offset : self._offset + 1]
            dst += 1
        if self.have_position:
            res[dst] = 1.0  # have position =True
            res[dst + 1] = (self._cur_close() - self.open_price) / self.open_price

        return res


class StocksEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)

    def __init__(
        self,
        prices,  # stock prices for instruments
        bars_count=DEFAULT_BARS_COUNT,  # count of bars we pass in obs
        commission=DEFAULT_COMMISSION_PERC,  # broker commission
        reset_on_close=True,  # episode stop on every close or sell of share
        state_1d=False,  # data-> h,l,c,v,h,l,c,v | or in 2d stack matrix h l c v |\n| h l c v
        random_ofs_on_reset=True,  # if true, on reset, start from random point rather than beginning
        reward_on_close=False,  # end of the episode or in every step
        volumes=False,  # to return the volumes
    ):

        self._prices = prices
        """               ___State___               """
        if state_1d:
            self._state = State1D(
                bars_count,
                commission,
                reset_on_close,
                reward_on_close=reward_on_close,
                volumes=volumes,
            )
        else:
            self._state = State(
                bars_count,
                commission,
                reset_on_close,
                reward_on_close=reward_on_close,
                volumes=volumes,
            )
        """               ___actions___               """
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        """               ___observation___               """
        self.observation_space = gym.spaces.Box(low=-np.inf, high=0)

        self.random_ofs_on_reset = random_ofs_on_reset
        self._seed()

    def reset(self):
        # make selection of financial intrument and it's offset
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        # offset
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0] - bars * 10) + bars
        else:
            offset = bars  # else from initial postion

        # reset the state
        self._state.reset(prices, offset)

        return self._state.encode()  # state in numpy array

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
