import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import random

# 常数
K_1 = 0.5
K_2 = 0.5

def randomDate(start, end, format):
    stime = datetime.strptime(start, format)
    etime = datetime.strptime(end, format)
    prop = random.random()
    ptime = stime + prop * (etime - stime)
    return ptime.strftime(format)

def preProcessData(file):
    """
    对读取raw data, 并进行初步整理
    """
    df=pd.read_csv(file,parse_dates=True,index_col=0)
    df=df.reset_index()
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    return df

def episode(df, date):
    # filter df by day
    df['datetime'] = pd.to_datetime(df['datetime'])
    filtered_df = df[df['datetime'].dt.date == pd.to_datetime(date).date()]
    # 显示结果 
    output_columns = ['open', 'close', 'high', 'low', 'volume']
    return filtered_df[output_columns]

def store_date(df):
    """
    df -> ["2019-12-30", .....] 这样的日期
    """
    formatted_list = df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    # 去重
    unique_list = list(set(formatted_list))
    # 随机打乱
    random.shuffle(unique_list)
    return unique_list

class FuturesTradingEnv(gym.Env):
    def __init__(self, win_len, data_file_path, furtures):
        # furtures: ["IC", "IF"]
        super(FuturesTradingEnv, self).__init__()

        if furtures=="IC":
            self.contract_multiplier, self.min_security = 200, 0.2 # 每点200元, 最小变动价位0.2点 from http://www.cffex.com.cn/zz500/
        elif furtures=="IF":
            self.contract_multiplier, self.min_security = 300, 0.2 # 每点300元, 最小变动价位0.2点 from http://www.cffex.com.cn/hs300/
        # 观测历史价格的窗口宽度
        self.win_len = win_len
        
        # 初始化代码
        # 设置action_space和observation_space

        self.trans_fee=2.3*(10**(-5))
        self.slip = 0.2  # constant slippage
        
        # actions 有两个，long，short
        
        self.market_data = preProcessData(data_file_path)
        self.date_list = store_date(self.market_data)
        # print(self.date_list)
        self.date_idx = 0
        self.tot_date = len(self.date_list)
        # observation: 过去的market data，Indicator(BuyLine, SellLine), account_profit
        # self.observation_space = spaces.Dict(
        #     {
        #         "market_data": spaces.Box(low=-np.inf, high=np.inf, shape=(self.win_len, 5), dtype=np.float32), # 5: open, close, high, low, volume
        #         "indicators": spaces.Box(low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32),
        #         # "account": spaces.Box(low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32) 
        #     }
        # )
        self.observation_space = (self.win_len, 5)
        self.action_space = spaces.Discrete(2)
        # self.observation_space = episode(df=self.market_data, date="2019-12-30")[0: self.win_len] # 选一天的示例，计算observation_space
        # print(type(self.observation_space))
        # print(self.observation_space)
    def _init_account(self):
        principal = 5e5 # 本金
        return {
            "margin": principal,
            "principal": principal,
            "hold_float": 0, 
            "step_hold_profit": 0, 
            "profit": 0,
            "position": 0, 
        }
    def step(self, action):
        # 处理一步交易逻辑
        # action：{long 0, short 1} -> 论文中为{-1, 1}
        if action==0:
            action = -1
        # 返回observation, reward, done, info
        # self.t = t 
        
        # o_{t}, a_t (根据0~t-1的历史做出决定，在t实现action) -> r_t (在t实现a_t 交易行为，产生的account profit)
        self.trade(action) # update self.account_info
        info = self.account_info
        
        reward = self.DSR_reward(action)
        self.t += 1 # 顺序很重要
        observation = self.market_data_day[self.t - self.win_len: self.t] 
        done = (self.t == self.len)
        truncated = done
        print(info.values())
        return observation, reward, done, truncated, info

    def reset(self, date_idx=0):
        self.date_idx += 1 # 下一次reset 随机到另一天的数据
        # 模型输入需要观察wein_len 宽的历史数据，因此开票之后需要等一等
        self.t = self.win_len
        self.market_data_day = episode(df=self.market_data, 
                                       # randomDate(start, end, format)
                                       date=self.date_list[date_idx]
                                      ) 
        # print("market_data_day", self.market_data_day)
        self.market_obs = self.market_data_day[self.t - self.win_len: self.t] # [0, t-1] ['open', 'close', 'high', 'low', 'volume']
        HH, LC = self.market_obs['high'].max(), self.market_obs['close'].min()
        HC, LL = self.market_obs['close'].max(), self.market_obs['low'].min()
        range = max(HH - LC, HC - LL)
        # print(type(self.market_obs["open"]))
        open = self.market_obs["open"].iloc[-1]
        # BuyLine, SellLine
        self.indicators = [open + K_1 * range, open - K_2 * range]
        # print("market_obs", self.market_obs)
        self.len = len(self.market_data_day)
        # 初始化账号信息
        self.account_info = self._init_account()
        # 初始化At0, Bt0, 为计算differential Sharpe radio
        self.At0, self.Bt0 = 0, 0
        # 返回初始观察值, account_info
        observation = {
            "market_data": self.market_obs, # 5: open, close, high, low, volume
            "indicators": self.indicators
        }
        return observation["market_data"], self.account_info

    def render(self, mode='human'):
        # 可选：渲染环境状态
        pass

    def close(self):
        # 可选：清理工作
        pass
    def trade(self, action):
        """
        参考https://github.com/nctu-dcs-lab/iRDPG-for-Quantitative-Trading-on-Stock-Index-Futures/blob/main/environment.py中的trading 函数，模拟交易过程
        """
        if action==0:
            print('trading_action is zero, which is wrong.')
            raise AssertionError('trading_action is zero')
            
        pc_t, pc_t_ = self.market_data_day.iloc[self.t].loc["close"], self.market_data_day.iloc[self.t-1].loc["close"] # p_t, p_t-1
        info = self.account_info
        margin, principal, hold_float, step_hold_profit, profit, position = info['margin'], info['principal'], info['hold_float'], info['step_hold_profit'], info['profit'], info['position']

        '''### 依「交易訊號&持單狀態」執行交易rules ###'''
        if action == 1:  #action為交易訊號：{1=long, -1=short}
            if position == 1:  #position為agent下單狀態：{-1=空單, 0=無單, 1=多單}
                profit = 0
                step_hold_profit = (pc_t - pc_t_)*position
                hold_float += step_hold_profit
                margin += hold_float*self.contract_multiplier
            elif position == 0: 
                profit = 0
                #long->hold, margin值不会马上变化
                position = 1
            elif position ==-1:
                step_hold_profit= ((pc_t-pc_t_)*position - 2*self.slip) \
                    - self.trans_fee*np.abs(position - action)*pc_t
                profit = step_hold_profit + hold_float  # TODO: 和论文不一样，和参考仓库一样
                margin += hold_float*self.contract_multiplier
                hold_float = 0  #change position, 归零
                position = 1
                     
        elif action == -1:
            if position == 1:
                # self.step_hold_profit = self.profit
                step_hold_profit = ((pc_t-pc_t_)*position - 2*self.slip) \
                    - self.trans_fee*np.abs(position-action)*pc_t
                profit = step_hold_profit + hold_float
                hold_float = 0
                margin += hold_float*self.contract_multiplier
                position = -1
            elif position == 0:
                profit = 0
                # short后，hold,margin值不会马上变化
                position = -1
            elif position ==-1:
                profit = 0
                step_hold_profit = (pc_t-pc_t_)*position
                hold_float += step_hold_profit
                margin += hold_float*self.contract_multiplier
        self.account_info = {
            "margin": margin,
            "principal": principal,
            "hold_float": hold_float, 
            "step_hold_profit": step_hold_profit, 
            "profit": profit,
            "position": position, 
        }
    def DSR_reward(self, action):
        """
        参考https://github.com/nctu-dcs-lab/iRDPG-for-Quantitative-Trading-on-Stock-Index-Futures/blob/main/environment.py中的DSR_reward2 函数，
        计算differential Sharpe ratio 作为reward(即时奖励)
        """
        eta = 1/self.t # decay, 也可尝试使用常数
        # eta = 1/240
        info = self.account_info
        margin, principal, hold_float, step_hold_profit, profit, position = info.values()
        pc_t, pc_t_ = self.market_data_day.iloc[self.t].loc["close"], self.market_data_day.iloc[self.t-1].loc["close"] # p_t, p_t-1
        ### Calculate step return #####
        r_t = (pc_t-pc_t_ - 2*self.slip)*position - self.trans_fee*np.abs(position - action)*pc_t

        if self.Bt0 - self.At0**2 ==0:
            d_t = 0
        else:
            d_t = (self.Bt0*(r_t-self.At0)- 0.5*self.At0*(r_t**2-self.Bt0))/(self.Bt0 - self.At0**2)**1.5
        
        ### update At0 & Bt0 ###
        self.At0 = eta*r_t + (1-eta)*self.At0
        self.Bt0 = eta*r_t**2 + (1-eta)*self.Bt0
        return d_t
    
# # 使用环境的示例
# env = FuturesTradingEnv(win_len=30, data_file_path="./data/IC_2015to2018.csv", furtures="IC")
# observation = env.reset(env.date_idx)
# done = False
# import matplotlib.pyplot as plt
# reward_list = []
# profit_list = []
# while not done:
#     action = env.action_space.sample()  # 或者您的代理生成的行动
#     observation, reward, done, info = env.step(action)
#     reward_list.append(reward)
#     profit_list.append(info["profit"])
#     # print(observation.shape, reward, done, info)
#     env.render()
# env.close()
# plt.plot(reward_list, label="reward")
# plt.plot(profit_list, label="profit")

# plt.savefig("./reward.png")
# df = preProcessData("./data/IC_2015to2018.csv")
# print(type(df.iloc[0, 0]))
# df = episode(df, "2019-12-30")
# print(df)