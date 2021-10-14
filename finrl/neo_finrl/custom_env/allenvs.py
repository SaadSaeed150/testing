import numpy as np
import pandas as pd
from gym.utils import seeding
from stable_baselines3.common import utils
import gym
from gym import spaces
import matplotlib
from copy import deepcopy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import random


class StockTradingEnvStopLoss(gym.Env):
    """
    A stock trading environment for OpenAI gym
    This environment penalizes the model if excedeed the stop-loss threshold, selling assets with under expectation %profit, and also  
    for not maintaining a reserve of cash.
    This enables the model to do trading with high confidence and manage cash reserves in addition to performing trading procedures.

    Reward at any step is given as follows
        r_i = (sum(cash, asset_value) + additional_reward - total_penalty - initial_cash) / initial_cash / days_elapsed
        , where total_penalty = cash_penalty + stop_loss_penalty + low_profit_penalty
                cash_penalty = max(0, sum(cash, asset_value)*cash_penalty_proportion-cash)
                stop_loss_penalty = -1 * dot(holdings,negative_closing_diff_avg_buy)
                low_profit_penalty = -1 * dot(holdings,negative_profit_sell_diff_avg_buy)
                additional_reward = dot(holdings,positive_profit_sell_diff_avg_buy)

        This reward function takes into account a profit/loss ratio constraint, liquidity requirement, as well as long-term accrued rewards.
        This reward function also forces the model to trade only when it's really confident to do so.

    Parameters:
    state space: {start_cash, <owned_shares>, for s in stocks{<stock.values>}, }
        df (pandas.DataFrame): Dataframe containing data
        buy_cost_pct (float): cost for buying shares
        sell_cost_pct (float): cost for selling shares
        hmax (int): max number of share purchases allowed per asset
        discrete_actions (bool): option to choose whether perform dicretization on actions space or not
        shares_increment (int): multiples number of shares can be bought in each trade.
        stoploss_penalty (float): Maximum loss we can tolerate. Valid value range is between 0 and 1. If x is specified, then agent will force sell all holdings for a particular asset if current price < x * avg_buy_price 
        profit_loss_ratio (int, float): Expected profit/loss ratio. Only applicable when stoploss_penalty < 1.
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
        initial_amount: (int, float): Amount of cash initially available
        daily_information_columns (list(str)): Columns to use when building state space from the dataframe. It could be OHLC columns or any other variables such as technical indicators and turbulence index
        cash_penalty_proportion (int, float): Penalty to apply if the algorithm runs out of cash
        patient (bool): option to choose whether end the cycle when we're running out of cash or just don't buy anything until we got additional cash 
    action space: <share_dollar_purchases>
    TODO:
        add holdings to memory
        move transactions to after the clip step.
    tests:
        after reset, static strategy should result in same metrics
        given no change in prices, no change in asset values
    """
    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        date_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        stoploss_penalty=0.9,
        profit_loss_ratio=2,
        turbulence_threshold=None,
        print_verbosity=10,
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
        log_data = True,
        seed_value = 0
    ):
        self.df = df
        self.dt = df.copy()        
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.updated_holdings = np.zeros(len(df.loc[0].tic.unique()))
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.stoploss_penalty = stoploss_penalty
        self.min_profit_penalty  = 1 + profit_loss_ratio * (1 - self.stoploss_penalty) 
        self.turbulence_threshold = turbulence_threshold
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
        self.episode = -1  # initialize so we can call reset
        self.episode_history = []
        self.printed_header = not log_data
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.cash_penalty_proportion = cash_penalty_proportion
        self.logger = utils.configure_logger()
        self.terminal_state = False
        self.my_cash = initial_amount
        self.seed_value = seed_value
        if self.cache_indicator_data:
            #print("caching data")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            #print("data cached!")
    def seed(self, seed=0):
        seed = self.seed_value
        if seed is None:
            seed = int(round(time.time() * 1000))
        #print("seed is ->", seed)            
        random.seed(seed)
    @property
    def current_step(self):
        return self.date_index - self.starting_point
    def reset(self):
        if self.terminal_state == False  :
            self.seed()
            self.sum_trades = 0
            self.actual_num_trades = 0
            self.closing_diff_avg_buy = np.zeros(len(self.assets))
            self.profit_sell_diff_avg_buy = np.zeros(len(self.assets))
            self.n_buys = np.zeros(len(self.assets))
            self.avg_buy_price = np.zeros(len(self.assets))
            if self.random_start:
                starting_point = random.choice(range(int(len(self.dates) * 0.5)))
                self.starting_point = starting_point
            else:
                self.starting_point = 0
            self.date_index = self.starting_point
            self.turbulence = 0
            self.episode += 1
            self.actions_memory = []
            self.transaction_memory = []
            self.state_memory = []
            self.account_information = {
                "cash": [],
                "asset_value": [],
                "total_assets": [],
                "reward": [],
            }
            init_state = np.array(
                [self.initial_amount]
                + [0] * len(self.assets)
                + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(init_state)
            return init_state
        else:
            pass
    def get_date_vector(self, date, cols=None):
        if (cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            v = []
            for a in self.assets:
                subset = trunc_df[trunc_df[self.stock_col] == a]
                v += subset.loc[date, cols].tolist()
            assert len(v) == len(self.assets) * len(cols)
            return v
    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        if self.printed_header is False:
            self.log_step(reason=reason, terminal_reward=reward)
            # Add outputs to logger interface

            gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
            self.logger.record("environment/GainLoss_pct",(gl_pct - 1)*100)
            self.logger.record(
                "environment/total_assets",
                int(self.account_information["total_assets"][-1]),
            )
            reward_pct = self.account_information["total_assets"][-1] / self.initial_amount
            self.logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
            self.logger.record("environment/total_trades", self.sum_trades)
            self.logger.record(
                "environment/actual_num_trades",
                self.actual_num_trades,
            )
            self.logger.record(
                "environment/avg_daily_trades",
                self.sum_trades / (self.current_step),
            )
            self.logger.record(
                "environment/avg_daily_trades_per_asset",
                self.sum_trades / (self.current_step) / len(self.assets),
            )
            self.logger.record("environment/completed_steps", self.current_step)
            self.logger.record(
                "environment/sum_rewards", np.sum(self.account_information["reward"])
            )
            self.logger.record(
                "environment/cash_proportion",
                self.account_information["cash"][-1]
                / self.account_information["total_assets"][-1],
            )
        return state, reward, True, {}
    def log_step(self, reason, terminal_reward=None):
        
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        cash_pct = (
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{cash_pct*100:0.2f}%",
        ]
        self.episode_history.append(rec)
        if self.printed_header is False:
        
            print(self.template.format(*rec))
        
    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"  # column widths: 8, 10, 15, 7, 10
        print(
            self.template.format(
                "EPISODE",
                "STEPS",
                "TERMINAL_REASON",
                "CASH",
                "TOT_ASSETS",
                "TERMINAL_REWARD_unsc",
                "GAINLOSS_PCT",
                "CASH_PROPORTION",
            )
        )
        self.printed_header = True
    def get_reward(self):
        if self.current_step == 0:
            return 0
        else:
            total_assets = self.account_information["total_assets"][-1]
            cash = self.account_information["cash"][-1]
            holdings = self.state_memory[-1][1 : len(self.assets) + 1]
            neg_closing_diff_avg_buy = np.clip(self.closing_diff_avg_buy, -np.inf, 0)
            neg_profit_sell_diff_avg_buy = np.clip(self.profit_sell_diff_avg_buy, -np.inf, 0)
            pos_profit_sell_diff_avg_buy = np.clip(self.profit_sell_diff_avg_buy, 0, np.inf)

            cash_penalty = max(0, (total_assets * self.cash_penalty_proportion - cash))
            if self.current_step > 1:
                prev_holdings = self.state_memory[-2][1 : len(self.assets) + 1]
                stop_loss_penalty = -1 * np.dot(np.array(prev_holdings),neg_closing_diff_avg_buy)
            else:
                stop_loss_penalty = 0
            low_profit_penalty = -1 * np.dot(np.array(holdings),neg_profit_sell_diff_avg_buy)
            total_penalty = cash_penalty + stop_loss_penalty + low_profit_penalty
            
            additional_reward = np.dot(np.array(holdings),pos_profit_sell_diff_avg_buy)

            reward = ((total_assets - total_penalty + additional_reward) / self.initial_amount) - 1
            reward /= self.current_step 

            return reward
        
    def find_valid_actions(self, actions):
        #print("====  Date: ",self.dt.loc[self.date_index]['date'].values[0])
        closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))
        begin_cash = self.my_cash
        act = actions.copy()

        act = act * self.hmax
        
        holdings = self.updated_holdings
        # print("holdings before: ", holdings)
        if type(holdings) is list:
            holdings = np.array(holdings)
        # holdings = holdings * self.hmax
        # if self.date_index < 50 or self.date_index > 200: 
        #   print("Cash: ", begin_cash)
        #   print("Actions: ", act)
        #   print("holdings: ", holdings)
        #   print("holdings + actions: ", holdings + act)
        
        neg = np.where((holdings + act)<0)
        # print("Neg indexes:",neg)
        if len(neg)>0 :
            #print("YES NEG")
            tot = holdings + act
            act[tot<0] = 0
        # if self.date_index < 50 or self.date_index > 200: 
            
        #   print("Now buying/Selling: ", (closings * act).sum() , " rem: ", begin_cash - (closings * act).sum() )
            
        if (closings * act).sum() >  begin_cash :
            # print("====== greater then cash =========== ")
            act = act - act
        begin_cash =  begin_cash - (closings * act).sum()
        # if self.date_index < 50 or self.date_index > 200: 
        #   print("After Actions:", act)

        self.my_cash = begin_cash 
        self.updated_holdings = holdings + act
        asset_val = (self.updated_holdings*closings).sum()
        # print("updated holdings:",self.updated_holdings )
        # print("asset value:",asset_val)
        t_asset = asset_val + begin_cash
        return act,begin_cash,asset_val,t_asset
                
    def step(self, actions):
        # let's just log what we're doing in terms of max actions at each step.
        self.sum_trades += np.sum(np.abs(actions))
        # print header only first time
        if self.printed_header is False:
            self.log_header()
        # print if it's time.
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        # if we're at the end
        if self.date_index == len(self.dates) - 1:
            self.terminal_state = True
            # if we hit the end, set reward to total gains (or losses)
            return self.return_terminal(reward=self.get_reward())
        else:
#             print(actions)
#             print("==============")
            # compute value of cash + assets
            begin_cash = self.state_memory[-1][0]
            holdings = self.state_memory[-1][1 : len(self.assets) + 1]
            assert min(holdings) >= 0
            closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))
            # print("------")  
            if type(holdings) is list:
                  holdings = np.array(holdings)
            holdings = holdings*self.hmax
            
            # print("closings:", closings)
            # print("holdings:", holdings)

            asset_value = np.dot(holdings, closings)
            # reward is (cash + assets) - (cash_last_step + assets_last_step)
            reward = self.get_reward()
            # log the values of cash, assets, and total assets

            holdings = holdings // self.hmax
            # multiply action values by our scalar multiplier and save
            actions = actions * self.hmax
            # buy/sell only if the price is > 0 (no missing data in this particular date)
            actions = np.where(closings > 0, actions, 0)
            if self.turbulence_threshold is not None:
                # if turbulence goes over threshold, just clear out all positions
                if self.turbulence >= self.turbulence_threshold:
                    actions = -(np.array(holdings) * closings)
                    self.log_step(reason="TURBULENCE")
            # scale cash purchases to asset
            if self.discrete_actions:
                # convert into integer because we can't buy fraction of shares
                actions = np.where(closings > 0, actions // closings, 0)
                actions = actions.astype(int)
                # round down actions to the nearest multiplies of shares_increment
                actions = np.where(actions >= 0,
                                (actions // self.shares_increment) * self.shares_increment,
                                ((actions + self.shares_increment) // self.shares_increment) * self.shares_increment)
            else:
                actions = np.where(closings > 0, actions / closings, 0)

            # clip actions so we can't sell more assets than we hold
            actions = np.maximum(actions, -np.array(holdings))
            
            self.closing_diff_avg_buy = closings - (self.stoploss_penalty * self.avg_buy_price)
            if begin_cash >= self.stoploss_penalty * self.initial_amount:
                # clear out position if stop-loss criteria is met
                actions = np.where(self.closing_diff_avg_buy < 0, -np.array(holdings), actions)
                
                if any(np.clip(self.closing_diff_avg_buy, -np.inf, 0) < 0):
                    self.log_step(reason="STOP LOSS")

            # compute our proceeds from sells, and add to cash
            sells = -np.clip(actions, -np.inf, 0)
            proceeds = np.dot(sells, closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds
            # compute the cost of our buys
            buys = np.clip(actions, 0, np.inf)
            spend = np.dot(buys, closings)
            costs += spend * self.buy_cost_pct
            # if we run out of cash...
            if (spend + costs) > coh:
                if self.patient:
                    # ... just don't buy anything until we got additional cash
                    self.log_step(reason="CASH SHORTAGE")
                    actions = np.where(actions>0,0,actions)
                    spend = 0
                    costs = 0
                else:
                    # ... end the cycle and penalize
                    return self.return_terminal(
                        reason="CASH SHORTAGE",reward=self.get_reward()
                    )

            self.transaction_memory.append(actions)  # capture what the model's could do

            # get profitable sell actions
            sell_closing_price = np.where(sells>0, closings, 0) #get closing price of assets that we sold
            profit_sell = np.where(sell_closing_price - self.avg_buy_price > 0, 1, 0) #mark the one which is profitable

            self.profit_sell_diff_avg_buy = np.where(profit_sell==1, 
                                                    closings - (self.min_profit_penalty * self.avg_buy_price),
                                                    0)
            
            if any(np.clip(self.profit_sell_diff_avg_buy, -np.inf, 0) < 0):
                self.log_step(reason="LOW PROFIT")
            else:
                if any(np.clip(self.profit_sell_diff_avg_buy, 0, np.inf) > 0):
                    self.log_step(reason="HIGH PROFIT")

            # verify we didn't do anything impossible here
            assert (spend + costs) <= coh
            
            #log actual total trades we did up to current step
            self.actual_num_trades = np.sum(np.abs(np.sign(actions)))
            
            # update our holdings
            coh = coh - spend - costs
#             print("actions passing stoploss:", actions)
#            self.actions_memory.append(actions * self.hmax)  # capture what the model's trying to do
            # print("Before holdings:",holdings )
            # print("before act:", actions)
            holdings_updated = holdings + actions
            # print("After holdings:", holdings_updated)
            # Update average buy price
            buys = np.sign(buys)
            self.n_buys += buys
            self.avg_buy_price = np.where(buys > 0, self.avg_buy_price + ((closings - self.avg_buy_price) / self.n_buys), self.avg_buy_price) #incremental average
            
            #set as zero when we don't have any holdings anymore
            self.n_buys = np.where(holdings_updated > 0, self.n_buys, 0)
            self.avg_buy_price = np.where(holdings_updated > 0, self.avg_buy_price, 0) 
            
            self.date_index += 1
            if self.turbulence_threshold is not None:
                self.turbulence = self.get_date_vector(
                    self.date_index, cols=["turbulence"]
                )[0]
                
#             print("holdings_updated: ", holdings_updated)

            # Update State
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)
            actions, b_cash, asset_value, t_asset = self.find_valid_actions(actions)

            self.actions_memory.append(actions)  # capture what the model's trying to do
            self.account_information["cash"].append(b_cash)
            self.account_information["asset_value"].append(asset_value)
            self.account_information["total_assets"].append(t_asset)
            self.account_information["reward"].append(reward)
            # print("ations now", actions)
            # print("ations afet hmax", actions)


            return state, reward, False, {}
        
        
    def get_sb_env(self):
        def get_self():
            return deepcopy(self)
        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs
    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)
        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs
    
    def save_asset_memory(self):
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]) :
            ]
            df_account_value = pd.DataFrame(self.account_information)
            df_account_value = df_account_value.rename(columns={'total_assets': 'account_value'})
            df_account_value = df_account_value[['date','account_value']]
            return df_account_value

    def save_action_memory(self):
        if self.current_step == 0:
            return None
        else:
            df_actions = pd.DataFrame({
                    "date": self.dates[-len(self.account_information["cash"]) :],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory,})
        
            dicx = []
            f_tick = self.dt.loc[0].tic.unique()
            for i in f_tick:
                dicx.append([])

            for i in range(0,len(df_actions)):
                lst = df_actions.iloc[i]['actions']
                g = 0

                for k in lst:
                    dicx[g].append(k)
                    g+=1
            df_act = pd.DataFrame(columns=f_tick)
            g =0
            for i in f_tick:
                df_act[i] = dicx[g]
                g+=1

            df_act['date'] = df_actions['date']
            df_actions = df_act 
            df_actions = df_actions[:-1]
            df_actions.index =  df_actions.date
            df_actions = df_actions.drop(columns = ['date'])
            return df_actions


class StockTradingEnvGetGain2(gym.Env):
    
    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        date_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        stoploss_penalty=0.9,
        profit_loss_ratio=2,
        turbulence_threshold=None,
        print_verbosity=10,
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
        upper_range = 0,
        lower_range = 0,
        gainlossratio = 0.5,
        log_data = True,
        seed_value = 0
    ):
        self.df = df
        self.dt = df.copy()        
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.updated_holdings = np.zeros(len(df.loc[0].tic.unique()))
        self.stoploss_penalty = stoploss_penalty
        self.min_profit_penalty  = 1 + profit_loss_ratio * (1 - self.stoploss_penalty) 
        self.turbulence_threshold = turbulence_threshold
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
        self.episode = -1  # initialize so we can call reset
        self.episode_history = []
        self.printed_header = not log_data
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.cash_penalty_proportion = cash_penalty_proportion
        self.logger = utils.configure_logger()
        self.terminal_state = False
        self.my_cash = initial_amount
        self.lower_range = lower_range
        self.upper_range = upper_range
        self.gainlossratio = gainlossratio
        self.seed_value = seed_value
        if self.cache_indicator_data:
            #print("caching data")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            #print("data cached!")
    def seed(self, seed=0):
        seed = self.seed_value
        if seed is None:
            seed = int(round(time.time() * 1000))
        #print("seed is ->", seed)            
        random.seed(seed)
    @property
    def current_step(self):
        return self.date_index - self.starting_point
    def reset(self):
        if self.terminal_state == False  :
            self.seed()
            self.sum_trades = 0
            self.actual_num_trades = 0
            self.closing_diff_avg_buy = np.zeros(len(self.assets))
            self.profit_sell_diff_avg_buy = np.zeros(len(self.assets))
            self.n_buys = np.zeros(len(self.assets))
            self.avg_buy_price = np.zeros(len(self.assets))
            if self.random_start:
                starting_point = random.choice(range(int(len(self.dates) * 0.5)))
                self.starting_point = starting_point
            else:
                self.starting_point = 0
            self.date_index = self.starting_point
            self.turbulence = 0
            self.episode += 1
            self.actions_memory = []
            self.transaction_memory = []
            self.state_memory = []
            self.account_information = {
                "cash": [],
                "asset_value": [],
                "total_assets": [],
                "reward": [],
            }
            init_state = np.array(
                [self.initial_amount]
                + [0] * len(self.assets)
                + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(init_state)
            return init_state
        else:
            pass
    def get_date_vector(self, date, cols=None):
        if (cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            v = []
            for a in self.assets:
                subset = trunc_df[trunc_df[self.stock_col] == a]
                v += subset.loc[date, cols].tolist()
            assert len(v) == len(self.assets) * len(cols)
            return v
    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        if self.printed_header is False:
            self.log_step(reason=reason, terminal_reward=reward)
            # Add outputs to logger interface

            gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
            self.logger.record("environment/GainLoss_pct",(gl_pct - 1)*100)
            self.logger.record(
                "environment/total_assets",
                int(self.account_information["total_assets"][-1]),
            )
            reward_pct = self.account_information["total_assets"][-1] / self.initial_amount
            self.logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
            self.logger.record("environment/total_trades", self.sum_trades)
            self.logger.record(
                "environment/actual_num_trades",
                self.actual_num_trades,
            )
            self.logger.record(
                "environment/avg_daily_trades",
                self.sum_trades / (self.current_step),
            )
            self.logger.record(
                "environment/avg_daily_trades_per_asset",
                self.sum_trades / (self.current_step) / len(self.assets),
            )
            self.logger.record("environment/completed_steps", self.current_step)
            self.logger.record(
                "environment/sum_rewards", np.sum(self.account_information["reward"])
            )
            self.logger.record(
                "environment/cash_proportion",
                self.account_information["cash"][-1]
                / self.account_information["total_assets"][-1],
            )
        return state, reward, True, {}
    def log_step(self, reason, terminal_reward=None):
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        cash_pct = (
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{cash_pct*100:0.2f}%",
        ]
        self.episode_history.append(rec)
        if self.printed_header is False:
            print(self.template.format(*rec))
       
    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"  # column widths: 8, 10, 15, 7, 10
        print(
            self.template.format(
                "EPISODE",
                "STEPS",
                "TERMINAL_REASON",
                "CASH",
                "TOT_ASSETS",
                "TERMINAL_REWARD_unsc",
                "GAINLOSS_PCT",
                "CASH_PROPORTION",
            )
        )
        self.printed_header = True
    def get_reward(self):
        if self.current_step <= 1:
            return 0
        else:
            
            # Total Asset on current day
            end_asset = self.account_information["total_assets"][-1]
            # Total Asset on day before current day
            begin_asset = self.account_information["total_assets"][-2]
            
            # find gain/loss percentage
            gain_loss_percentage = ((end_asset - begin_asset)/begin_asset)*100
            
            penalty = 0
            award = 0            
            
            # self.gainlossratio controls how much reward/penalty should be given with gain_loss_percentage ratio            
            # If gain percentage is greater then upper limit then award
            if gain_loss_percentage >= self.upper_range:               
                # Award is calculated as ::
                # Lets say we have 
                # Gain Percentage (gain_loss_percentage) = 9% 
                # gainlossratio = 0.5 
                # then 9 + 9 * 0.5 = 13.5 percent of total asset of current day will be added to total_asset 
                # This reward will be scaled by dividing by initial amount and current steps ahead
                
                award = end_asset * (gain_loss_percentage + gain_loss_percentage * self.gainlossratio)
            elif gain_loss_percentage <= -self.lower_range :
                gain_loss_percentage = -(gain_loss_percentage)
                
                # Penalty is calculated as ::
                # Lets say we have 
                # Loss percentage (gain_loss_percentage) = 5% 
                # gainlossratio = 0.5 
                # then 5 + 5 * 0.5 = 7.5 percent of total asset of current day will be deducted from total_asset 
                # This reward will be scaled by dividing by initial amount and current steps ahead
                
                penalty = end_asset * (gain_loss_percentage + gain_loss_percentage * self.gainlossratio)


            # Scaling reward for model better learning 
            reward = ((end_asset + award - penalty) / self.initial_amount) - 1           
            reward /= self.current_step 
            
            return reward
    def find_valid_actions(self, actions):
        # print("====  Date: ",self.dt.loc[self.date_index]['date'].values[0])
        closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))
        begin_cash = self.my_cash
        act = actions.copy()

        act = act * self.hmax
        
        holdings = self.updated_holdings
        # print("holdings before: ", holdings)
        if type(holdings) is list:
            holdings = np.array(holdings)
        # holdings = holdings * self.hmax
        # if self.date_index < 50 or self.date_index > 200: 
        #   print("Cash: ", begin_cash)
        #   print("Actions: ", act)
        #   print("holdings: ", holdings)
        #   print("holdings + actions: ", holdings + act)
        
        neg = np.where((holdings + act)<0)
        # print("Neg indexes:",neg)
        if len(neg)>0 :
            #print("YES NEG")
            tot = holdings + act
            act[tot<0] = 0
        # if self.date_index < 50 or self.date_index > 200: 
            
        #   print("Now buying/Selling: ", (closings * act).sum() , " rem: ", begin_cash - (closings * act).sum() )
            
        if (closings * act).sum() >  begin_cash :
            # print("====== greater then cash =========== ")
            act = act - act
        begin_cash =  begin_cash - (closings * act).sum()
        # if self.date_index < 50 or self.date_index > 200: 
        #   print("After Actions:", act)

        self.my_cash = begin_cash 
        self.updated_holdings = holdings + act
        asset_val = (self.updated_holdings*closings).sum()
        # print("updated holdings:",self.updated_holdings )
        # print("asset value:",asset_val)
        t_asset = asset_val + begin_cash
        return act,begin_cash,asset_val,t_asset

    def step(self, actions):
        # let's just log what we're doing in terms of max actions at each step.
        self.sum_trades += np.sum(np.abs(actions))
        # print header only first time
        if self.printed_header is False:
            self.log_header()
        # print if it's time.
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        # if we're at the end
        if self.date_index == len(self.dates) - 1:
            self.terminal_state = True
            # if we hit the end, set reward to total gains (or losses)
            return self.return_terminal(reward=self.get_reward())
        else:
            # compute value of cash + assets
            begin_cash = self.state_memory[-1][0]
            holdings = self.state_memory[-1][1 : len(self.assets) + 1]
            assert min(holdings) >= 0
            closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))

            if type(holdings) is list:
                  holdings = np.array(holdings)
            holdings = holdings*self.hmax
            
            asset_value = np.dot(holdings, closings)
            # reward is (cash + assets) - (cash_last_step + assets_last_step)
            reward = self.get_reward()
            holdings = holdings // self.hmax            
             
            # multiply action values by our scalar multiplier and save
            actions = actions * self.hmax
            # buy/sell only if the price is > 0 (no missing data in this particular date)
            actions = np.where(closings > 0, actions, 0)
            if self.turbulence_threshold is not None:
                # if turbulence goes over threshold, just clear out all positions
                if self.turbulence >= self.turbulence_threshold:
                    actions = -(np.array(holdings) * closings)
                    self.log_step(reason="TURBULENCE")
            # scale cash purchases to asset
            if self.discrete_actions:
                # convert into integer because we can't buy fraction of shares
                actions = np.where(closings > 0, actions // closings, 0)
                actions = actions.astype(int)
                # round down actions to the nearest multiplies of shares_increment
                actions = np.where(actions >= 0,
                                (actions // self.shares_increment) * self.shares_increment,
                                ((actions + self.shares_increment) // self.shares_increment) * self.shares_increment)
            else:
                actions = np.where(closings > 0, actions / closings, 0)

            # clip actions so we can't sell more assets than we hold
            actions = np.maximum(actions, -np.array(holdings))
            
            self.closing_diff_avg_buy = closings - (self.stoploss_penalty * self.avg_buy_price)
            if begin_cash >= self.stoploss_penalty * self.initial_amount:
                # clear out position if stop-loss criteria is met
                actions = np.where(self.closing_diff_avg_buy < 0, -np.array(holdings), actions)
                

            # compute our proceeds from sells, and add to cash
            sells = -np.clip(actions, -np.inf, 0)
            proceeds = np.dot(sells, closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds
            # compute the cost of our buys
            buys = np.clip(actions, 0, np.inf)
            spend = np.dot(buys, closings)
            costs += spend * self.buy_cost_pct
            # if we run out of cash...
            if (spend + costs) > coh:
                if self.patient:
                    # ... just don't buy anything until we got additional cash
                    actions = np.where(actions>0,0,actions)
                    spend = 0
                    costs = 0
                else:
                    # ... end the cycle and penalize
                    return self.return_terminal(
                        reward=self.get_reward()
                    )

            self.transaction_memory.append(actions)  # capture what the model's could do

            # get profitable sell actions
            sell_closing_price = np.where(sells>0, closings, 0) #get closing price of assets that we sold
            profit_sell = np.where(sell_closing_price - self.avg_buy_price > 0, 1, 0) #mark the one which is profitable

            self.profit_sell_diff_avg_buy = np.where(profit_sell==1, 
                                                    closings - (self.min_profit_penalty * self.avg_buy_price),
                                                    0)
            

            # verify we didn't do anything impossible here
            assert (spend + costs) <= coh
            
            #log actual total trades we did up to current step
            self.actual_num_trades = np.sum(np.abs(np.sign(actions)))
            
            # update our holdings
            coh = coh - spend - costs

            
            holdings_updated = holdings + actions

            # Update average buy price
            buys = np.sign(buys)
            self.n_buys += buys
            self.avg_buy_price = np.where(buys > 0, self.avg_buy_price + ((closings - self.avg_buy_price) / self.n_buys), self.avg_buy_price) #incremental average
            
            #set as zero when we don't have any holdings anymore
            self.n_buys = np.where(holdings_updated > 0, self.n_buys, 0)
            self.avg_buy_price = np.where(holdings_updated > 0, self.avg_buy_price, 0) 
            
            self.date_index += 1
            if self.turbulence_threshold is not None:
                self.turbulence = self.get_date_vector(
                    self.date_index, cols=["turbulence"]
                )[0]

            # Update State
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)
            actions, b_cash, asset_value, t_asset = self.find_valid_actions(actions)

            self.actions_memory.append(actions)  # capture what the model's trying to do
            self.account_information["cash"].append(b_cash)
            self.account_information["asset_value"].append(asset_value)
            self.account_information["total_assets"].append(t_asset)
            self.account_information["reward"].append(reward)

            return state, reward, False, {}
    def get_sb_env(self):
        def get_self():
            return deepcopy(self)
        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs
    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)
        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs
    
    def save_asset_memory(self):
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]) :
            ]
            df_account_value = pd.DataFrame(self.account_information)
            df_account_value = df_account_value.rename(columns={'total_assets': 'account_value'})
            df_account_value = df_account_value[['date','account_value']]

            return df_account_value

    def save_action_memory(self):
        if self.current_step == 0:
            return None
        else:
            df_actions = pd.DataFrame({
                    "date": self.dates[-len(self.account_information["cash"]) :],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory,})
        
            dicx = []
            f_tick = self.dt.loc[0].tic.unique()
            for i in f_tick:
                dicx.append([])

            for i in range(0,len(df_actions)):
                lst = df_actions.iloc[i]['actions']
                g = 0

                for k in lst:
                    dicx[g].append(k)
                    g+=1
            df_act = pd.DataFrame(columns=f_tick)
            g =0
            for i in f_tick:
                df_act[i] = dicx[g]
                g+=1

            df_act['date'] = df_actions['date']
            df_actions = df_act 
            df_actions = df_actions[:-1]
            df_actions.index =  df_actions.date
            df_actions = df_actions.drop(columns = ['date'])
            return df_actions
        
class StockTradingEnvCashpenalty(gym.Env):
    """
    A stock trading environment for OpenAI gym
    This environment penalizes the model for not maintaining a reserve of cash.
    This enables the model to manage cash reserves in addition to performing trading procedures.
    Reward at any step is given as follows
        r_i = (sum(cash, asset_value) - initial_cash - max(0, sum(cash, asset_value)*cash_penalty_proportion-cash))/(days_elapsed)
        This reward function takes into account a liquidity requirement, as well as long-term accrued rewards.
    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        buy_cost_pct (float): cost for buying shares
        sell_cost_pct (float): cost for selling shares
        hmax (int, array): maximum cash to be traded in each trade per asset. If an array is provided, then each index correspond to each asset
        discrete_actions (bool): option to choose whether perform dicretization on actions space or not
        shares_increment (int): multiples number of shares can be bought in each trade. Only applicable if discrete_actions=True
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
        initial_amount: (int, float): Amount of cash initially available
        daily_information_columns (list(str)): Columns to use when building state space from the dataframe. It could be OHLC columns or any other variables such as technical indicators and turbulence index
        cash_penalty_proportion (int, float): Penalty to apply if the algorithm runs out of cash
        patient (bool): option to choose whether end the cycle when we're running out of cash or just don't buy anything until we got additional cash

    RL Inputs and Outputs
        action space: [<n_assets>,] in range {-1, 1}
        state space: {start_cash, [shares_i for in in assets], [[indicator_j for j in indicators] for i in assets]]}
    TODO:
        Organize functions
        Write README
        Document tests
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        date_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        turbulence_threshold=None,
        print_verbosity=10,
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
        log_data = True,
        seed_value = 0
    ):
        self.df = df
        self.dt = df.copy()
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.updated_holdings = np.zeros(len(df.loc[0].tic.unique()))
        self.my_cash = initial_amount
        self.df = self.df.set_index(date_col_name)
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.turbulence_threshold = turbulence_threshold
        self.daily_information_cols = daily_information_cols
        self.seed_value = seed_value
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
        self.episode = -1  # initialize so we can call reset
        self.episode_history = []
        self.printed_header = not log_data
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.logger = utils.configure_logger()

        self.cash_penalty_proportion = cash_penalty_proportion
        if self.cache_indicator_data:
            #print("caching data")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            #print("data cached!")

    def seed(self, seed=0):
        seed = self.seed_value
        if seed is None:
            seed = int(round(time.time() * 1000))
        #print("seed is ->", seed)            
        random.seed(seed)

    @property
    def current_step(self):
        return self.date_index - self.starting_point

    @property
    def cash_on_hand(self):
        # amount of cash held at current timestep
        return self.state_memory[-1][0]

    @property
    def holdings(self):
        # Quantity of shares held at current timestep
        return self.state_memory[-1][1 : len(self.assets) + 1]

    @property
    def closings(self):
        return np.array(self.get_date_vector(self.date_index, cols=["close"]))

    def reset(self):
        self.seed()
        self.sum_trades = 0
        if self.random_start:
            starting_point = random.choice(range(int(len(self.dates) * 0.5)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": [],
        }
        init_state = np.array(
            [self.initial_amount]
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )
        self.state_memory.append(init_state)
        return init_state

    def get_date_vector(self, date, cols=None):
        if (cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            v = []
            for a in self.assets:
                subset = trunc_df[trunc_df[self.stock_col] == a]
                v += subset.loc[date, cols].tolist()
            assert len(v) == len(self.assets) * len(cols)
            return v

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        if self.printed_header is False:
            self.log_step(reason=reason, terminal_reward=reward)
            # Add outputs to logger interface
            gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
            self.logger.record("environment/GainLoss_pct", (gl_pct - 1) * 100)
            self.logger.record(
                "environment/total_assets",
                int(self.account_information["total_assets"][-1]),
            )
            reward_pct = self.account_information["total_assets"][-1] / self.initial_amount
            self.logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
            self.logger.record("environment/total_trades", self.sum_trades)
            self.logger.record(
                "environment/avg_daily_trades",
                self.sum_trades / (self.current_step),
            )
            self.logger.record(
                "environment/avg_daily_trades_per_asset",
                self.sum_trades / (self.current_step) / len(self.assets),
            )
            self.logger.record("environment/completed_steps", self.current_step)
            self.logger.record(
                "environment/sum_rewards", np.sum(self.account_information["reward"])
            )
            self.logger.record(
                "environment/cash_proportion",
                self.account_information["cash"][-1]
                / self.account_information["total_assets"][-1],
            )
        return state, reward, True, {}

    def log_step(self, reason, terminal_reward=None):
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        cash_pct = (
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{cash_pct*100:0.2f}%",
        ]
        self.episode_history.append(rec)
        if self.printed_header is False:
        
            print(self.template.format(*rec))

    def log_header(self):
        if self.printed_header is False:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"  # column widths: 8, 10, 15, 7, 10
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "TERMINAL_REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "TERMINAL_REWARD_unsc",
                    "GAINLOSS_PCT",
                    "CASH_PROPORTION",
                )
            )
            self.printed_header = True

    def get_reward(self):
        if self.current_step == 0:
            return 0
        else:
            assets = self.account_information["total_assets"][-1]
            cash = self.account_information["cash"][-1]
            cash_penalty = max(0, (assets * self.cash_penalty_proportion - cash))
            assets -= cash_penalty
            reward = (assets / self.initial_amount) - 1
            reward /= self.current_step
            return reward

    def get_transactions(self, actions):
        """
        This function takes in a raw 'action' from the model and makes it into realistic transactions
        This function includes logic for discretizing
        It also includes turbulence logic.
        """
        # record actions of the model

        # multiply actions by the hmax value
        actions = actions * self.hmax

        # Do nothing for shares with zero value
        actions = np.where(self.closings > 0, actions, 0)

        # discretize optionally
        if self.discrete_actions:
            # convert into integer because we can't buy fraction of shares
            actions = actions // self.closings
            actions = actions.astype(int)
            # round down actions to the nearest multiplies of shares_increment
            actions = np.where(
                actions >= 0,
                (actions // self.shares_increment) * self.shares_increment,
                ((actions + self.shares_increment) // self.shares_increment)
                * self.shares_increment,
            )
        else:
            actions = actions / self.closings

        # can't sell more than we have
        actions = np.maximum(actions, -np.array(self.holdings))

        # deal with turbulence
        if self.turbulence_threshold is not None:
            # if turbulence goes over threshold, just clear out all positions
            if self.turbulence >= self.turbulence_threshold:
                actions = -(np.array(self.holdings))
                self.log_step(reason="TURBULENCE")

        return actions
    def find_valid_actions(self, actions):
        # print("====  Date: ",self.dt.loc[self.date_index]['date'].values[0])
        closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))
        begin_cash = self.my_cash
        act = actions.copy()

        act = act * self.hmax
        
        holdings = self.updated_holdings
        # print("holdings before: ", holdings)
        if type(holdings) is list:
            holdings = np.array(holdings)
        # holdings = holdings * self.hmax
        # if self.date_index < 50 or self.date_index > 200: 
        #   print("Cash: ", begin_cash)
        #   print("Actions: ", act)
        #   print("holdings: ", holdings)
        #   print("holdings + actions: ", holdings + act)
        
        neg = np.where((holdings + act)<0)
        # print("Neg indexes:",neg)
        if len(neg)>0 :
            #print("YES NEG")
            tot = holdings + act
            act[tot<0] = 0
        # if self.date_index < 50 or self.date_index > 200: 
            
        #   print("Now buying/Selling: ", (closings * act).sum() , " rem: ", begin_cash - (closings * act).sum() )
            
        if (closings * act).sum() >  begin_cash :
            # print("====== greater then cash =========== ")
            act = act - act
        begin_cash =  begin_cash - (closings * act).sum()
        # if self.date_index < 50 or self.date_index > 200: 
        #   print("After Actions:", act)

        self.my_cash = begin_cash 
        self.updated_holdings = holdings + act
        asset_val = (self.updated_holdings*closings).sum()
        # print("updated holdings:",self.updated_holdings )
        # print("asset value:",asset_val)
        t_asset = asset_val + begin_cash
        return act,begin_cash,asset_val,t_asset

    def step(self, actions):
        # let's just log what we're doing in terms of max actions at each step.
        self.sum_trades += np.sum(np.abs(actions))
        self.log_header()
        # print if it's time.
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        # if we're at the end
        if self.date_index == len(self.dates) - 1:
            # if we hit the end, set reward to total gains (or losses)
            return self.return_terminal(reward=self.get_reward())
        else:
            """
            First, we need to compute values of holdings, save these, and log everything.
            Then we can reward our model for its earnings.
            """
            # compute value of cash + assets
            begin_cash = self.cash_on_hand
            assert min(self.holdings) >= 0
            asset_value = np.dot(self.holdings, self.closings)
            # log the values of cash, assets, and total assets

            # compute reward once we've computed the value of things!
            reward = self.get_reward()

            """
            Now, let's get down to business at hand. 
            """
            transactions = self.get_transactions(actions)

            # compute our proceeds from sells, and add to cash
            sells = -np.clip(transactions, -np.inf, 0)
            proceeds = np.dot(sells, self.closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds
            # compute the cost of our buys
            buys = np.clip(transactions, 0, np.inf)
            spend = np.dot(buys, self.closings)
            costs += spend * self.buy_cost_pct
            # if we run out of cash...
            if (spend + costs) > coh:
                if self.patient:
                    # ... just don't buy anything until we got additional cash
                    self.log_step(reason="CASH SHORTAGE")
                    transactions = np.where(transactions > 0, 0, transactions)
                    spend = 0
                    costs = 0
                else:
                    # ... end the cycle and penalize
                    return self.return_terminal(
                        reason="CASH SHORTAGE", reward=self.get_reward()
                    )
            self.transaction_memory.append(
                transactions
            )  # capture what the model's could do
            # verify we didn't do anything impossible here
            assert (spend + costs) <= coh
            # update our holdings
            coh = coh - spend - costs

            holdings_updated = self.holdings + transactions
            self.date_index += 1
            if self.turbulence_threshold is not None:
                self.turbulence = self.get_date_vector(
                    self.date_index, cols=["turbulence"]
                )[0]
            # Update State
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)
            actions, b_cash, asset_value, t_asset = self.find_valid_actions(actions)

            self.actions_memory.append(actions)  # capture what the model's trying to do
            self.account_information["cash"].append(b_cash)
            self.account_information["asset_value"].append(asset_value)
            self.account_information["total_assets"].append(t_asset)
            self.account_information["reward"].append(reward)

            return state, reward, False, {}

    def get_sb_env(self):
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)

        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]) :
            ]
            df_account_value = pd.DataFrame(self.account_information)
            df_account_value = df_account_value.rename(columns={'total_assets': 'account_value'})
            df_account_value = df_account_value[['date','account_value']]
            return df_account_value

    def save_action_memory(self):
        if self.current_step == 0:
            return None
        else:
            df_actions = pd.DataFrame({
                    "date": self.dates[-len(self.account_information["cash"]) :],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory,})
        
            dicx = []
            f_tick = self.dt.loc[0].tic.unique()
            for i in f_tick:
                dicx.append([])

            for i in range(0,len(df_actions)):
                lst = df_actions.iloc[i]['actions']
                g = 0

                for k in lst:
                    dicx[g].append(k)
                    g+=1
            df_act = pd.DataFrame(columns=f_tick)
            g =0
            for i in f_tick:
                df_act[i] = dicx[g]
                g+=1

            df_act['date'] = df_actions['date']
            df_actions = df_act 
            df_actions = df_actions[:-1]
            df_actions.index =  df_actions.date
            df_actions = df_actions.drop(columns = ['date'])
            return df_actions
        
class StockTradingEnvGetGain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df, 
                stock_dim,
                hmax,                
                initial_amount,
                buy_cost_pct,
                sell_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                risk_indicator_col='turbulence',
                make_plots = False, 
                print_verbosity = 10,
                day = 0, 
                initial=True,
                previous_state=[],
                model_name = '',
                mode='',
                iteration='',
                days_notrade = 0,
                upper_range = 0,
                lower_range = 0,
                seed_value = 0
                ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list+ ['close_30_sma', 'boll_lb'] 
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False          
        self.seed_value = seed_value
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name=model_name
        self.mode=mode 
        self.iteration=iteration
        self.state = self._initiate_state()
        self.trades = 0
        self.episode = 0
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        self._seed()
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.status = {}
        self.total_days = {}
        
        for i in train.tic.unique():
            self.status[i] = 1
            self.total_days[i] = days_notrade
            
            
        self.test_data = df.copy()        
        self.days_notrade = days_notrade 
        self.upper_range = upper_range
        self.lower_range= lower_range
                       

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index+1]>0: 
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index+self.stock_dim+1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action),self.state[index+self.stock_dim+1])
                    sell_amount = self.state[index+1] * sell_num_shares 
                    #update balance
                    self.state[0] += sell_amount

                    self.state[index+self.stock_dim+1] -= sell_num_shares
                    self.cost +=self.state[index+1] * sell_num_shares 
                    self.trades+=1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares
            
        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence>=self.turbulence_threshold:
                if self.state[index+1]>0: 
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions 
                    if self.state[index+self.stock_dim+1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index+self.stock_dim+1]
                        sell_amount = self.state[index+1]*sell_num_shares
                        #update balance
                        self.state[0] += sell_amount
                        self.state[index+self.stock_dim+1] =0
                        self.cost += self.state[index+1]*self.state[index+self.stock_dim+1]
                        self.trades+=1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    
    def _buy_stock(self, index, action):

        def _do_buy():
            if self.state[index+1]>0: 
                #Buy only if the price is > 0 (no missing data in this particular date)       
                available_amount = self.state[0] // self.state[index+1]
                # print('available_amount:{}'.format(available_amount))
                
                #update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = self.state[index+1] * buy_num_shares 
                self.state[0] -= buy_amount
#                 print ("cash after buying:",self.state[0] , " buy amount:",buy_amount)
                self.state[index+self.stock_dim+1] += buy_num_shares
                
                self.cost+=self.state[index+1] * buy_num_shares 
                self.trades+=1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence< self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory,'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def step(self, actions):
    
        self.terminal = self.day >= len(self.df.index.unique())-1
        
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()            
            end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = self.state[0]+sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))- self.initial_amount 
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return']=df_total_value['account_value'].pct_change(1)
            if df_total_value['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name!='') and (self.mode!=''):
                df_actions = self.save_action_memory()
                df_actions.to_csv('results/actions_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration))
                df_total_value.to_csv('results/account_value_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                df_rewards.to_csv('results/account_rewards_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.plot(self.asset_memory,'r')
                plt.savefig('results/account_value_{}_{}_{}.png'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.close()

            return self.state, self.reward, self.terminal, {}

        else:
#             print("self close:", self.test_data.iloc[self.day]['close'])
#             print("self.state:",self.state)
#             print("divide:", self.test_data.iloc[self.day]['close']/self.state)
            if self.upper_range!=0 and self.lower_range!=0 and self.days_notrade!=0 :
        
                lists_act = []
                ix = 0
#                 print("===================")
#                 print("Stocks list: ",self.state[self.stock_dim+1: 30+self.stock_dim+1])
                
                for tick in self.test_data.tic.unique():

                    close_ = self.test_data.loc[self.day][self.test_data.loc[self.day]['tic'] == tick]['close'].values

                    change = ((close_/self.status[tick])-1)* 100 
                    self.total_days[tick] +=1
                    
                    if(self.total_days[tick] > self.days_notrade and (change > self.upper_range or change < -self.lower_range)):
                        self.total_days[tick] = 0
                        if self.state[ix+self.stock_dim+1] > 0:
#                             print("Selling "+ str(self.state[ix+self.stock_dim+1])+" of "+tick+" index: "+ str(ix)+ " at day:",self.df.loc[self.day:].date.values[0] \
#                                  ," change: ",change, " close last trade: ",self.status[tick], " close today:",close_ )

                            actions[ix] = - self.state[ix+self.stock_dim+1]
                        self.status[tick] = close_ 
                    elif self.total_days[tick] > self.days_notrade:
                        self.total_days[tick] = 0
                        self.status[tick] = close_                        
                    
                    else:
                        actions[ix] = 0.0
                    ix+=1
            #print("Stks:",self.state[self.stock_dim+1:29+self.stock_dim+1])
#             print("\n")

            actions = actions * self.hmax #actions initially is scaled between 0 to 1
            actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-self.hmax]*self.stock_dim)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)
#             print("============== Date: "+self.df.loc[self.day:].date.values[0]+" =========")
            
#             print("cash hand:", self.state[0])            
#             print("stock value:", sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])))

#             print("\n")
            
            #state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day,:]    
            if self.turbulence_threshold is not None:     
                self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state =  self._update_state()
                           
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        #initiate state
        self.state = self._initiate_state()
        
        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        
        self.episode+=1

        return self.state
    
    def render(self, mode='human',close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique())>1:
                # for multiple stock
                state = [self.initial_amount] + \
                         self.data.close.values.tolist() + \
                         [0]*self.stock_dim  + \
                         sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            else:
                # for single stock
                state = [self.initial_amount] + \
                        [self.data.close] + \
                        [0]*self.stock_dim  + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        else:
            #Using Previous State
            if len(self.df.tic.unique())>1:
                # for multiple stock
                state = [self.previous_state[0]] + \
                         self.data.close.values.tolist() + \
                         self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                         sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            else:
                # for single stock
                state = [self.previous_state[0]] + \
                        [self.data.close] + \
                        self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        return state

    def _update_state(self):
        if len(self.df.tic.unique())>1:
            # for multiple stock
            state =  [self.state[0]] + \
                      self.data.close.values.tolist() + \
                      list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])

        else:
            # for single stock
            state =  [self.state[0]] + \
                     [self.data.close] + \
                     list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                     sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
                          
        return state

    def _get_date(self):
        if len(self.df.tic.unique())>1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        #print("THIS function is called")
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique())>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=0):
        seed = self.seed_value
        #print("seed is ->", seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_multiproc_env(self, n = 10):
        def get_self():
            return deepcopy(self)
        e = SubprocVecEnv([get_self for _ in range(n)], start_method = 'fork')
        obs = e.reset()
        return e, obs

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
        

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df, 
                stock_dim,
                hmax,                
                initial_amount,
                buy_cost_pct,
                sell_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                risk_indicator_col='turbulence',
                make_plots = False, 
                print_verbosity = 10,
                day = 0, 
                initial=True,
                previous_state=[],
                model_name = '',
                mode='',
                iteration='',
                seed_value = 0):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list+ ['close_30_sma', 'boll_lb'] 
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name=model_name
        self.mode=mode 
        self.iteration=iteration
        # initalize state
        self.state = self._initiate_state()
        self.seed_value = seed_value
        
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        #self.reset()
        self._seed()
        


    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index+1]>0: 
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index+self.stock_dim+1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action),self.state[index+self.stock_dim+1])
                    sell_amount = self.state[index+1] * sell_num_shares * (1- self.sell_cost_pct)
                    #update balance
                    self.state[0] += sell_amount

                    self.state[index+self.stock_dim+1] -= sell_num_shares
                    self.cost +=self.state[index+1] * sell_num_shares * self.sell_cost_pct
                    self.trades+=1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares
            
        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence>=self.turbulence_threshold:
                if self.state[index+1]>0: 
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions 
                    if self.state[index+self.stock_dim+1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index+self.stock_dim+1]
                        sell_amount = self.state[index+1]*sell_num_shares* (1- self.sell_cost_pct)
                        #update balance
                        self.state[0] += sell_amount
                        self.state[index+self.stock_dim+1] =0
                        self.cost += self.state[index+1]*self.state[index+self.stock_dim+1]* \
                                    self.sell_cost_pct
                        self.trades+=1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    
    def _buy_stock(self, index, action):

        def _do_buy():
            if self.state[index+1]>0: 
                #Buy only if the price is > 0 (no missing data in this particular date)       
                available_amount = self.state[0] // self.state[index+1]
                # print('available_amount:{}'.format(available_amount))
                
                #update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = self.state[index+1] * buy_num_shares * (1+ self.buy_cost_pct)
                self.state[0] -= buy_amount

                self.state[index+self.stock_dim+1] += buy_num_shares
                
                self.cost+=self.state[index+1] * buy_num_shares * self.buy_cost_pct
                self.trades+=1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence< self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory,'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()            
            end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = self.state[0]+sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))- self.initial_amount 
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return']=df_total_value['account_value'].pct_change(1)
            if df_total_value['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name!='') and (self.mode!=''):
                df_actions = self.save_action_memory()
                df_actions.to_csv('results/actions_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration))
                df_total_value.to_csv('results/account_value_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                df_rewards.to_csv('results/account_rewards_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.plot(self.asset_memory,'r')
                plt.savefig('results/account_value_{}_{}_{}.png'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.close()

            # Add outputs to logger interface
            #logger.record("environment/portfolio_value", end_total_asset)
            #logger.record("environment/total_reward", tot_reward)
            #logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            #logger.record("environment/total_cost", self.cost)
            #logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}

        else:

            actions = actions * self.hmax #actions initially is scaled between 0 to 1
            actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-self.hmax]*self.stock_dim)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)
            
            #state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day,:]    
            if self.turbulence_threshold is not None:     
                self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state =  self._update_state()
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        #initiate state
        self.state = self._initiate_state()
        
        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        
        self.episode+=1

        return self.state
    
    def render(self, mode='human',close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique())>1:
                # for multiple stock
                state = [self.initial_amount] + \
                         self.data.close.values.tolist() + \
                         [0]*self.stock_dim  + \
                         sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            else:
                # for single stock
                state = [self.initial_amount] + \
                        [self.data.close] + \
                        [0]*self.stock_dim  + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        else:
            #Using Previous State
            if len(self.df.tic.unique())>1:
                # for multiple stock
                state = [self.previous_state[0]] + \
                         self.data.close.values.tolist() + \
                         self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                         sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            else:
                # for single stock
                state = [self.previous_state[0]] + \
                        [self.data.close] + \
                        self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        return state

    def _update_state(self):
        if len(self.df.tic.unique())>1:
            # for multiple stock
            state =  [self.state[0]] + \
                      self.data.close.values.tolist() + \
                      list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])

        else:
            # for single stock
            state =  [self.state[0]] + \
                     [self.data.close] + \
                     list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                     sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
                          
        return state

    def _get_date(self):
        if len(self.df.tic.unique())>1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        print("THIS function is called")
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique())>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=0):
        seed = self.seed_value
        #print("seed is ->", seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_multiproc_env(self, n = 10):
        def get_self():
            return deepcopy(self)
        e = SubprocVecEnv([get_self for _ in range(n)], start_method = 'fork')
        obs = e.reset()
        return e, obs

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
