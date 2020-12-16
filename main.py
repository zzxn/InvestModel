import math
import os
from typing import List

import numpy as np
import pandas as pd

from config import config

entry_count = 0

col_names = ['code', 'name', 'pe_dynamic', 'pe_ttm', 'pe_static',
             'roe15', 'roe16', 'roe17', 'roe18', 'reo19',
             'cash_net_incr15', 'cash_net_incr16', 'cash_net_incr17', 'cash_net_incr18', 'cash_net_incr19',
             'op_profit15', 'op_profit16', 'op_profit17', 'op_profit18', 'op_profit19',
             'cur_lia15', 'cur_lia16', 'cur_lia17', 'cur_lia18', 'cur_lia19',
             'total_assets15', 'total_assets16', 'total_assets17', 'total_assets18', 'total_assets19']


class StringIndexArrayAccessor:
    def __init__(self, data: np.ndarray, col_names: List[str]):
        assert data.shape[1] == len(col_names)
        self.name_index = {name: i for i, name in enumerate(col_names)}
        self.data = data

    def __getitem__(self, cols):
        assert len(cols) > 0
        if len(cols) == 1:
            return self.data.T[self.name_index[cols[0]]]
        else:
            return self.data.T[[self.name_index[c] for c in cols]]


def read_table(path: str, col_names: List[str]) -> StringIndexArrayAccessor:
    table = pd.read_excel(path)
    data = []
    for line in table.values:
        if type(line[0]) == float and math.isnan(line[0]):
            break
        flag = False
        # for i in range(len(line)):
        #     if line[i] == '——':
        #         flag = True
        if not flag:
            data.append(line)
    global entry_count
    print("{}: {} entries".format(path.split('.')[-2], len(data)))
    entry_count += len(data)
    data = np.stack(data)
    for line in data:
        for i in range(len(line)):
            if line[i] == '——':
                line[i] = 1e-10
    return StringIndexArrayAccessor(data, col_names)


def compute_index(accessor: StringIndexArrayAccessor, model_type: str):
    roe: np.ndarray = accessor['roe15', 'roe16', 'roe17', 'roe18', 'reo19']  # [5, N]
    # N = roe.shape[1]
    # roe = np.array([[roe[i, j] if type(roe[i, j]) == float else 0.0 for j in range(N)] for i in range(5)])  # remove str
    roe_avg = np.average(roe, axis=0)
    roe_index = roe_avg / np.max(roe_avg)

    pes: np.ndarray = accessor['pe_ttm', 'pe_dynamic', 'pe_static'].astype('float64')  # [3, N]
    pe_weights: np.ndarray = np.array([config[f'model.{model_type}.pe_ratio.sub_weight.ttm'],
                                       config[f'model.{model_type}.pe_ratio.sub_weight.dynamic'],
                                       config[f'model.{model_type}.pe_ratio.sub_weight.static']],
                                      dtype='float64').reshape([1, -1])  # [1, 3]

    weight_pe: np.ndarray = (pe_weights @ pes).flatten()  # [N]
    pe_index = -weight_pe / np.max(weight_pe)

    cash_net_incr = accessor['cash_net_incr15', 'cash_net_incr16', 'cash_net_incr17',
                             'cash_net_incr18', 'cash_net_incr19'].astype('float64')  # [5, N]
    op_profit = accessor['op_profit15', 'op_profit16', 'op_profit17', 'op_profit18', 'op_profit19'].astype(
        'float64')  # [5, N]

    cash_flow = cash_net_incr / op_profit  # [5, N]
    cash_flow_index = np.average(cash_flow, axis=0)

    cur_lia = accessor['cur_lia15', 'cur_lia16', 'cur_lia17', 'cur_lia18', 'cur_lia19'].astype('float64')  # [5, N]
    total_assets = accessor['total_assets15', 'total_assets16', 'total_assets17',
                            'total_assets18', 'total_assets19'].astype('float64')  # [5, N]

    debt_rate = cur_lia / total_assets
    debt_index = -np.average(debt_rate, axis=0)

    total_index = pe_index * config[f'model.{model_type}.pe_ratio.weight'] \
                  + cash_flow_index * config[f'model.{model_type}.cash_flow.weight'] \
                  + debt_index * config[f'model.{model_type}.debt_ratio.weight'] \
                  + roe_index * config[f'model.{model_type}.roe.weight']

    total_index_bias = np.min(roe, axis=0) < config[f'model.{model_type}.roe.min']
    total_index_bias = total_index_bias.astype('float64')
    total_index_bias *= -1000.0

    total_index += total_index_bias

    return (weight_pe, cash_flow.T, debt_rate.T, roe_avg,
            pe_index, cash_flow_index, debt_index, roe_index, total_index)


def process_table(table_path, out_path, model_type):
    a = read_table(table_path, col_names)
    i = compute_index(a, model_type)  # M * [N * ?]
    i = [index if (len(index.shape) > 1) else np.expand_dims(index, axis=-1) for index in i]
    augment_data = np.concatenate(i, axis=1)
    augment_data_col_names = ['weight_pe',
                              'cash_flow15', 'cash_flow16', 'cash_flow17', 'cash_flow18', 'cash_flow19',
                              'debt_rate15', 'debt_rate16', 'debt_rate17', 'debt_rate18', 'debt_rate19',
                              'roe_avg',
                              'pe_index', 'cash_flow_index', 'debt_index', 'roe_index', 'total_index']

    final_data = np.concatenate((a.data, augment_data), axis=1)
    final_col_names = col_names + augment_data_col_names

    def pe_filter(row: np.ndarray):
        pe = row[final_col_names.index('weight_pe')]  # TODO: assure it
        return pe >= config[f'model.{model_type}.pe_ratio.min']

    final_data = [row for row in final_data if pe_filter(row)]
    final_data.sort(key=lambda row: row[-1], reverse=True)
    final_data = np.stack(final_data)
    final_df = pd.DataFrame(data=final_data, columns=final_col_names)

    final_df.to_excel(out_path)


def main():
    infile_names = os.listdir('./resources')
    outfile_names = [os.path.splitext(name) for name in infile_names]
    outfile_names = [name[0] + '.out' + name[1] for name in outfile_names]
    print(infile_names)
    print(outfile_names)
    for infile, outfile in zip(infile_names, outfile_names):
        process_table(f'./resources/{infile}', f'./out/{outfile}', 'normal')

    print("Processed {} entries.".format(entry_count))


if __name__ == '__main__':
    main()
