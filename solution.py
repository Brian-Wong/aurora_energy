import argparse
import math
import itertools
from time import time
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from pulp import *

class Model:
    def __init__(self, args):
        self.args = args
        self.open_files()
        self.daily_idx = self.daily_data.index
        self.half_hourly_idx = self.half_hourly_data.index

    def open_files(self):
        half_hourly_data = pd.read_excel(self.args.data_fp, sheet_name='Half-hourly data', parse_dates=['Unnamed: 0'])
        daily_data = pd.read_excel(self.args.data_fp, sheet_name='Daily data', parse_dates=['Unnamed: 0'])
        params = pd.read_excel(self.args.params_fp)

        half_hourly_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
        half_hourly_data.set_index(half_hourly_data.columns[0], inplace=True)
        daily_data.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
        daily_data.set_index(daily_data.columns[0], inplace=True)
        params.rename(columns={'Unnamed: 0': 'Variable'}, inplace=True)
        params.set_index(params.columns[0], inplace=True)

        half_hourly_data = pd.merge(half_hourly_data, daily_data, left_index=True, right_index=True, how='left')
        half_hourly_data.ffill(inplace=True)

        half_hourly_data = half_hourly_data.sort_index().loc[args.start_datetime:args.end_datetime,]
        daily_data = daily_data.sort_index().loc[args.start_datetime:args.end_datetime,]

        params = params['Values'].to_dict()
        params['init_volume'] = self.args.init_volume
        params['init_cycles'] = self.args.init_cycles
        params['Storage volume degradation rate'] /= 100

        self.half_hourly_data = half_hourly_data
        self.daily_data = daily_data
        self.params = params

    def calc_depreciation(self):
        depreciation_per_cycle = self.params['Capex'] / self.params['Lifetime (2)']
        # assume charge and discharge wears out battery equally
        depreciation_per_MW = depreciation_per_cycle / (2 * self.params['Max storage volume']) 
        depreciation_per_MW_per_timeslot = depreciation_per_MW / 2 # divide by 2 because each timeslot is 30min and unit for storage vol is MWh
        self.depreciation = depreciation_per_MW_per_timeslot

    def create_bs_prices(self, df, cols):
        self.calc_depreciation()
        new_df = df[cols]
        new_df['min'] = np.amin(new_df, axis=1)
        new_df['max'] = np.amax(new_df, axis=1)
        buy_price = new_df['min'] * (1 + self.params['Battery charging efficiency']) + self.depreciation
        buy_price = buy_price.to_dict()
        sell_price = new_df['max'] * (1 - self.params['Battery charging efficiency']) - self.depreciation
        sell_price = sell_price.to_dict()

        return buy_price, sell_price
    
    def set_objectives(self):
        # objective
        self.obj  = lpSum([-self.half_hourly_buy[i] * self.half_hourly_buy_price[i] for i in self.half_hourly_idx]) * 0.5
        self.obj += lpSum([-self.daily_buy[i] * self.daily_buy_price[i] for i in self.daily_idx]) * 24
        self.obj += lpSum([-self.half_hourly_sell[i] * self.half_hourly_sell_price[i] for i in self.half_hourly_idx]) * 0.5
        self.obj += lpSum([-self.daily_sell[i] * self.daily_sell_price[i] for i in self.daily_idx]) * 24
        self.model += self.obj
    
    def set_constraints(self):
        self.list_constraints = []
        # constraints
        self.prev_volume = self.params['init_volume']
        self.init_volume = self.params['init_volume']
        self.cycles = self.params['init_cycles']
        self.max_capacity = self.params['Max storage volume']
        self.current_capacity = [self.max_capacity] * len(self.half_hourly_idx)
        self.degradation_rate = self.params['Storage volume degradation rate']
        self.max_cycles = self.params['Lifetime (2)']
        self.list_total_buys = [self.half_hourly_buy[k] + self.daily_buy[pd.Timestamp(k.date())] for k in self.half_hourly_idx]
        self.list_total_sells = [self.half_hourly_sell[k] + self.daily_sell[pd.Timestamp(k.date())] for k in self.half_hourly_idx]
        self.list_gross_charge = [(a - b) / 2 for a, b in zip(self.list_total_buys, self.list_total_sells)]
        self.list_net_charge = [(a + b) / 2 for a, b in zip(self.list_total_buys, self.list_total_sells)]
        #current_volume = np.cumsum([prev_volume] + list_net_charge)[1:]
        self.list_cycles = [gross_charge / (2 * self.max_capacity) for gross_charge in self.list_gross_charge] 
        # degrade battery
        #current_capacity = [max_capacity * (1 - cycles * degradation_rate) for cycles in list_cycles]
        #for i in range(len(self.half_hourly_idx)):
        #    self.model += self.list_total_buys[i] <= self.params['Max charging rate']
        #    self.model += self.list_total_sells[i] >= -self.params['Max discharging rate']
            #self.model += list_cycles[i] <= max_cycles
            #self.model += current_volume[i] <= current_capacity[i]
            #self.model += current_volume[i] >= 0
            #self.model += lpSum([self.list_net_charge[i] for i in range(len(self.half_hourly_idx))]) <= self.max_capacity#current_capacity[i]
            #self.model += lpSum([self.list_net_charge[i] for i in range(len(self.half_hourly_idx))]) >= 0
        self.model += lpSum(self.list_cycles) <= self.max_cycles
        self.model += lpSum(self.list_net_charge) + self.init_volume >= 0
        self.model += lpSum(self.list_net_charge) + self.init_volume <= self.max_capacity
        self.model += lpSum(self.list_net_charge) - self.init_volume == 0 #total charge + total discharge - initial charge == 0
        # method 2
        prev_min = 0 - self.prev_volume
        prev_max = self.max_capacity - self.prev_volume
        n = self.args.n # break the problem into 2-hourly chunks
        for i in range(0, len(self.half_hourly_idx), n):
            self.prev_volume = lpSum(self.list_net_charge[i:i+n])
            self.model += self.prev_volume >= prev_min
            self.model += self.prev_volume <= prev_max
            prev_min -= self.prev_volume
            prev_max -= self.prev_volume
        # method 1
        #n = self.args.n
        #for i in range(0, len(self.half_hourly_idx), n):
        ##    print(i)
        #    self.model += lpSum(self.list_net_charge[:i]) + self.init_volume >= 0
        #    self.model += lpSum(self.list_net_charge[:i]) + self.init_volume <= self.max_capacity
    
    def solve_model(self, solver=None):
        if not solver:
            self.model.solve()
        else:
            self.model.solve(solver)

    def print_results(self):
        print(f'Status: {LpStatus[self.model.status]}')
        print(f'P/L: {self.model.objective.value():,}')
        
    def post_processing(self):
        # check number of batteries required for cycles used
        self.total_cycles = sum([x.value() for x in self.list_cycles])
        self.req_batteries = math.ceil(self.total_cycles / self.params['Lifetime (2)'])
        # reconstruct the dataframe
        self.post_df_half_hourly = pd.DataFrame(index=self.half_hourly_idx)
        self.post_df_daily = pd.DataFrame(index=self.daily_idx)
        self.post_df_half_hourly['half_hourly_buy'] = [self.half_hourly_buy[k].value() for k in self.half_hourly_idx]
        self.post_df_half_hourly['half_hourly_sell'] = [self.half_hourly_sell[k].value() for k in self.half_hourly_idx]
        self.post_df_daily['daily_buy'] = [self.daily_buy[k].value() for k in self.daily_idx]
        self.post_df_daily['daily_sell'] = [self.daily_sell[k].value() for k in self.daily_idx]
        self.post_df = pd.merge(self.post_df_half_hourly, self.post_df_daily, how='left', left_index=True, right_index=True)
        self.post_df.ffill(inplace=True)
        #self.post_df /= 2 # half because the buys / sells are in MWh for half hour slots
        #self.post_df['net_charge'] = np.sum(self.post_df, axis=1)
        self.post_df['net_charge'] = [x.value() for x in self.list_net_charge]
        self.post_df['rolling_charge'] = np.cumsum(self.post_df['net_charge']) 
        self.post_df['rolling_charge'] += self.params['init_volume']
        self.post_df = pd.merge(self.post_df, self.half_hourly_data, left_index=True, right_index=True)
        self.post_df['half_hourly_buy_price'] = self.half_hourly_buy_price
        self.post_df['half_hourly_sell_price'] = self.half_hourly_sell_price
        self.post_df = pd.merge(self.post_df, self.daily_data, left_index=True, right_index=True, how='left')
        self.post_df.ffill(inplace=True)
        self.post_df.to_csv(self.args.output_csv)
        print(self.post_df['rolling_charge'].max(), self.post_df['rolling_charge'].min())
        check_csv = self.post_df.resample('D').agg('last')
        check_csv.to_csv('check.csv')

    def build_model(self):
        self.model = LpProblem(name='Energy_Trading', sense=LpMaximize)
        
        # decision variables
        self.half_hourly_buy = LpVariable.dicts(name='Half_hourly_Buys', indices=self.half_hourly_idx, 
                                           lowBound=0, upBound=self.params['Max charging rate'])
        self.half_hourly_sell = LpVariable.dicts(name='Half_hourly_Sells', indices=self.half_hourly_idx, 
                                            lowBound=-self.params['Max discharging rate'], upBound=0)
        self.daily_buy = LpVariable.dicts(name='Daily_Buys', indices=self.daily_idx, 
                                     lowBound=0, upBound=self.params['Max charging rate'])
        self.daily_sell = LpVariable.dicts(name='Daily_Sells', indices=self.daily_idx, 
                                      lowBound=-self.params['Max discharging rate'], upBound=0)
        # get price dictionary
        # to simplify problem, decision variables are taken from perspective of battery
        # therefore, buy and sell prices are adjusted for efficiency rate
        self.half_hourly_buy_price, self.half_hourly_sell_price = self.create_bs_prices(df=self.half_hourly_data, cols=['Market 1 Price [£/MWh]', 'Market 2 Price [£/MWh]'])
        self.daily_buy_price, self.daily_sell_price = self.create_bs_prices(df=self.daily_data, cols=['Market 3 Price [£/MWh]'])

        self.set_objectives()
        self.set_constraints()
        self.solve_model()
        self.print_results()
        self.post_processing()

        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_fp', type=str, default='attachment.xlsx', help='Please input filepath to the data')
    parser.add_argument('-params_fp', type=str, default='constraints.xlsx', help='Please input filepath to the constraints / parameters')
    #parser.add_argument('-output_fp', type=str, default='output.txt', help='Please output filepath')
    parser.add_argument('-output_csv', type=str, default='output.csv', help='Please output csv filepath')
    parser.add_argument('-init_volume', type=float, default=0, help='Please input the initial volume of the battery in MW')
    parser.add_argument('-init_cycles', type=float, default=0, help='Please input the initial # cycles of the battery')
    parser.add_argument('-start_datetime', type=str, default='2018-01-01 00:00:00', help='Please input the start datetime for the problem')
    parser.add_argument('-end_datetime', type=str, default='2020-12-31 23:59:59', help='Please input the end datetime for the problem')
    parser.add_argument('-n', type=int, default= 2 * 24 * 15, help='Please input the time interval to check that battery charge is between 0 and max capacity, in # half hourly slots')
    #parser.add_argument('-end_datetime', type=str, default='2018-12-31 23:59:59', help='Please input the end datetime for the problem')
    args = parser.parse_args()
    t0 = time()
    model = Model(args)
    model.build_model()
    print(f'Time taken: {time()-t0}s')