#!/usr/bin/env python3

import argparse
import csv
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd

k=39


def transform(df: pd.DataFrame, clean: bool, basename: str) -> tuple[pd.DataFrame, str]:
    if clean == True:
        basename = basename[:-6]
        logging.info('Reading in previously-cleaned data')
        df_clean = read_in_cleaned_data(df)

    else:
        df_clean = massage_data(df)
        logging.info('Writing clean data to csv file')
        df_clean.to_csv(basename + '_clean.csv', encoding='utf-8', index=False)

    return(df_clean, basename)



def read_in_cleaned_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info('starting to format previously cleaned data')

    df_clean = df
    for index, row in df.iterrows():
        my_list = row['babbles'][1:-1].split(',')
        int_list = [eval(i) for i in my_list]
        df_clean.at[index, 'babbles'] = int_list

    logging.info('done formatting previously cleaned data')
    return df_clean



def massage_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info('starting to massage data')

    #keep = ['Bout ID (sans subtype)', 'Cluster39All']
    keep = ['Bout ID (sans subtype)', 'TREATMENT', 'Sex', 'Cluster6']
    df = df.filter(keep)

    # combine babbles from same Bout ID into list
    data = pd.DataFrame(columns=['boutID', 'treatment', 'sex', 'babbles'] )
    data['boutID'] = df['Bout ID (sans subtype)'].unique()
    data['babbles'] = [[] for _ in range(data.shape[0])]
    
    this_list=[]
    num=0
    for i, r in df.iterrows():
     
        this_list.append(r['Cluster6'])

        try:
            if (r['Bout ID (sans subtype)'] != df.at[i+1, 'Bout ID (sans subtype)']):
    
                logging.debug(f'{r["Bout ID (sans subtype)"]} != {df.at[i+1, "Bout ID (sans subtype)"]}')
                for ii, rr in data.iterrows():
                    if rr['boutID'] == r['Bout ID (sans subtype)']:
                        data['treatment'][ii] = r['TREATMENT']
                        #data['treatment_period'][ii] = r['Treatment_Period']
                        #data['age'][ii] = r['AGE']
                        data['sex'][ii] = r['Sex']
                        data['babbles'][ii] = this_list

                num+=1
                if (num%10==0):
                    logging.info(f'completed {num}/{data.shape[0]} bouts')
    
                this_list = []
    
            else:
                continue

        except KeyError:   # keyerror on last item in df

            for ii, rr in data.iterrows():
                if rr['boutID'] == r['Bout ID (sans subtype)']:
                    data['treatment'][ii] = r['TREATMENT']
                    #data['treatment_period'][ii] = r['Treatment_Period']
                    #data['age'][ii] = r['AGE']
                    data['sex'][ii] = r['Sex']
                    data['babbles'][ii] = this_list

            this_list = []

    logging.info('finished massaging data')

    return data



    """
    >>> data after filtering                                    
         Bout ID (sans subtype)  Cluster39All 
    0         18B26A_N2_43396_0            34 
    1         18B26A_N2_43396_0            34 
    2         18B26A_N2_43396_0            36 
    ...                     ...           ... 
    3296     18B26A_N2_43406_10            13 
    3297     18B26A_N2_43406_10            13 
    3298     18B26A_N2_43406_10            11 
                                              
    [3299 rows x 2 columns]                   
    """

    """
    >>> data after re-organzing                                    
                    boutID                                            babbles
    0    18B26A_N2_43396_0                                       [34, 34, 36]
    1    18B26A_N2_43396_1  [6, 12, 6, 36, 12, 23, 37, 34, 12, 12, 27, 21,...
    2    18B26A_N2_43396_2           [27, 6, 29, 10, 2, 6, 10, 2, 29, 39, 10]
    ..                 ...                                                ...
    97   18B26A_N2_43406_8  [10, 12, 23, 37, 30, 30, 10, 13, 6, 23, 30, 32...
    98   18B26A_N2_43406_9  [13, 24, 25, 36, 36, 34, 12, 36, 37, 27, 2, 13...
    99  18B26A_N2_43406_10  [21, 6, 6, 6, 37, 13, 23, 6, 13, 23, 35, 23, 2...
                                                                             
    [100 rows x 2 columns]                                                   
    """



def dump_bouts(df_clean: pd.DataFrame, minlength: int, dump: bool):
    logging.info('starting to dump individual bouts')

    counter=0
    for index, row in df_clean.iterrows():
        if ( len(row['babbles']) < minlength ):
            continue
        else:
            counter += 1
            logging.info('dumping a bout')
            df_small = df_clean[index:index+1]
            df_small.to_csv(f'dump_{counter}_clean.csv', encoding='utf-8', index=False)

    logging.info(f'{counter} bouts were greater than or equal to {minlength} signals')

    return



def analysis_singles(df_clean: pd.DataFrame, minlength: int, basename: str):
    singles = count_singles(df_clean, minlength)

    with open(basename + '_singles.json', 'w') as o:
        json.dump(singles, o, indent=2)
    with open(basename + '_singles.csv', 'w') as o:
        writer = csv.writer(o)
        writer.writerow(['signal', 'freq'])
        for item in singles.keys():
            writer.writerow([item, singles[item]])

    plot_singles(singles, basename, df_clean)

    return



def analysis_pairs(df_clean: pd.DataFrame, minlength: int, basename: str):
    pairs = count_pairs(df_clean, minlength)

    with open(basename + '_pairs.json', 'w') as o:
        json.dump(pairs, o, indent=2)
    with open(basename + '_pairs.csv', 'w') as o:
        writer = csv.writer(o)
        writer.writerow(['signal1', 'signal2', 'freq'])
        for itema in pairs.keys():
            for itemb in pairs[itema].keys():
                # writing signal as '1', '2', etc, instead of 'a1', 'b2', etc
                writer.writerow([itema[1:], itemb[1:], pairs[itema][itemb]])

    plot_pairs(pairs, basename, df_clean)

    return



def analysis_triples(df_clean: pd.DataFrame, minlength: int, basename: str):
    if minlength < 3:
        minlength = 3
        logging.info('setting minlength to 3 for triples analysis')

    triples = count_triples(df_clean, minlength)

    with open(basename + '_triples.json', 'w') as o:
        json.dump(triples, o, indent=2)
    with open(basename + '_triples.csv', 'w') as o:
        writer = csv.writer(o)
        writer.writerow(['signal1', 'signal2', 'signal3', 'freq'])
        for itema in triples.keys():
            for itemb in triples[itema].keys():
                for itemc in triples[itema][itemb].keys():
                    writer.writerow([itema[1:], itemb[1:], itemc[1:], triples[itema][itemb][itemc]])

    # First row of csv output is header, then contains Nx (NxN blocks)
    # So in the case of 39 signals, Rows 2-40 of output correspond to 
    # all bouts that start with signal 1, then is 39x39 matrix of second
    # and third signal. First look left for second signal, then go down
    # for third signal. Rows 41-79 would be the second 'block' - all 
    # bouts that start with signal 2, etc.
    return



def analysis_quads(df_clean: pd.DataFrame, minlength: int, basename: str):
    if minlength < 4:
        minlength = 4
        logging.info('setting minlength to 4 for quads analysis')

    quads = count_quads(df_clean, minlength)

    with open(basename + '_quads.json', 'w') as o:
        json.dump(quads, o, indent=2)
    with open(basename + '_quads.csv', 'w') as o:
        writer = csv.writer(o)
        writer.writerow(['signal1', 'signal2', 'signal3', 'signal4', 'freq'])
        for a in quads.keys():
            for b in quads[a].keys():
                for c in quads[a][b].keys():
                    for d in quads[a][b][c].keys():
                        writer.writerow([a[1:], b[1:], c[1:], d[1:], quads[a][b][c][d]])

    return



def analysis_quints(df_clean: pd.DataFrame, minlength: int, basename: str):
    if minlength < 5:
        minlength = 5
        logging.info('setting minlength to 5 for quints analysis')

    quints = count_quints(df_clean, minlength)

    with open(basename + '_quints.json', 'w') as o:
        json.dump(quints, o, indent=2)
    with open(basename + '_quints.csv', 'w') as o:
        writer = csv.writer(o)
        writer.writerow(['signal1', 'signal2', 'signal3', 'signal4', 'signal5', 'freq'])
        for a in quints.keys():
            for b in quints[a].keys():
                for c in quints[a][b].keys():
                    for d in quints[a][b][c].keys():
                        for e in quints[a][b][c][d].keys():
                            writer.writerow([a[1:], b[1:], c[1:], d[1:], e[1:], quints[a][b][c][d][e]])

    return



def count_singles(df_clean: pd.DataFrame, minlength: int) -> dict:
    logging.info('starting to count singles')

    freq_singles = {}
    for val in list(range(1,k+1)):
        freq_singles[val] = 0

    for index, row in df_clean.iterrows():
        if ( len(row['babbles']) < minlength):
            continue
        else:
            for item in row['babbles']:
                try:
                    freq_singles[item] += 1
                except KeyError as e: 
                    logging.debug(f'KeyError for {item}')
    
    logging.info('finished counting singles')

    return(freq_singles)



def count_pairs(df_clean: pd.DataFrame, minlength: int) -> dict:
    logging.info('starting to count pairs')

    freq_pairs = {}
    for vala in [ 'a'+str(i) for i in range(1,k+1) ]:
        freq_pairs[vala] = {}
        for valb in [ 'b'+str(i) for i in range(1,k+1) ]:
            freq_pairs[vala][valb] = 0

    counter=0
    for index, row in df_clean.iterrows():
        if ( len(row['babbles']) < minlength ):
            continue
        else:
            counter += 1
            for first, second in zip(row['babbles'], row['babbles'][1:]):
                try:
                    freq_pairs['a'+str(first)]['b'+str(second)] += 1
                except KeyError as e:
                    logging.debug(f'KeyError for {first} or {second}')

    logging.info(f'{counter} bouts were greater than or equal to {minlength} signals')
    logging.info('finished counting pairs')

    return(freq_pairs)



def count_triples(df_clean: pd.DataFrame, minlength: int) -> dict:
    logging.info('starting to count triples')

    freq_triples = {}
    for vala in [ 'a'+str(i) for i in range(1,k+1) ]:
        freq_triples[vala] = {}
        for valb in [ 'b'+str(i) for i in range(1,k+1) ]:
            freq_triples[vala][valb] = {}
            for valc in [ 'c'+str(i) for i in range(1,k+1) ]:
                freq_triples[vala][valb][valc] = 0

    counter=0
    for index, row in df_clean.iterrows():
        if ( len(row['babbles']) < minlength ):
            continue
        else:
            counter += 1
            for first, second, third in zip(row['babbles'], row['babbles'][1:], row['babbles'][2:]):
                try:
                    freq_triples['a'+str(first)]['b'+str(second)]['c'+str(third)] += 1
                except KeyError as e:
                    logging.debug(f'KeyError for {first} or {second} or {third}')

    logging.info(f'{counter} bouts were greater than or equal to {minlength} signals')
    logging.info('finished counting triples')

    return(freq_triples)



def count_quads(df_clean: pd.DataFrame, minlength: int) -> dict:
    logging.info('starting to count quads')

    freq_quads = {}
    for vala in [ 'a'+str(i) for i in range(1,k+1) ]:
        freq_quads[vala] = {}
        for valb in [ 'b'+str(i) for i in range(1,k+1) ]:
            freq_quads[vala][valb] = {}
            for valc in [ 'c'+str(i) for i in range(1,k+1) ]:
                freq_quads[vala][valb][valc] = {}
                for vald in [ 'd'+str(i) for i in range(1,k+1) ]:
                    freq_quads[vala][valb][valc][vald] = 0

    counter=0
    for index, row in df_clean.iterrows():
        if ( len(row['babbles']) < minlength ):
            continue
        else:
            counter += 1
            for first, second, third, fourth in zip(row['babbles'], row['babbles'][1:], row['babbles'][2:], row['babbles'][3:]):
                try:
                    freq_quads['a'+str(first)]['b'+str(second)]['c'+str(third)]['d'+str(fourth)] += 1
                except KeyError as e:
                    logging.debug(f'KeyError for {first} or {second} or {third} or {fourth}')

    logging.info(f'{counter} bouts were greater than or equal to {minlength} signals')
    logging.info('finished counting quads')

    return(freq_quads)



def count_quints(df_clean: pd.DataFrame, minlength: int) -> dict:
    logging.info('starting to count quints')

    freq_quints = {}
    for vala in [ 'a'+str(i) for i in range(1,k+1) ]:
        freq_quints[vala] = {}
        for valb in [ 'b'+str(i) for i in range(1,k+1) ]:
            freq_quints[vala][valb] = {}
            for valc in [ 'c'+str(i) for i in range(1,k+1) ]:
                freq_quints[vala][valb][valc] = {}
                for vald in [ 'd'+str(i) for i in range(1,k+1) ]:
                    freq_quints[vala][valb][valc][vald] = {}
                    for vale in [ 'e'+str(i) for i in range(1,k+1) ]:
                        freq_quints[vala][valb][valc][vald][vale] = 0

    counter=0
    for index, row in df_clean.iterrows():
        if ( len(row['babbles']) < minlength ):
            continue
        else:
            counter += 1
            for first, second, third, fourth, fifth in zip(row['babbles'], row['babbles'][1:], row['babbles'][2:], row['babbles'][3:], row['babbles'][4:]):
                try:
                    freq_quints['a'+str(first)]['b'+str(second)]['c'+str(third)]['d'+str(fourth)]['e'+str(fifth)] += 1
                except KeyError as e:
                    logging.debug(f'KeyError for {first} or {second} or {third} or {fourth} or {fifth}')

    logging.info(f'{counter} bouts were greater than or equal to {minlength} signals')
    logging.info('finished counting quints')

    return(freq_quints)



def plot_singles(data: dict, basename: str, dfc: pd.DataFrame):

    logging.info('plotting singles')

    # plot histogram for single sounds
    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values(), color='indianred')
    odd_labels =  [ i if i%2==1 else ' ' for i in range(1,k+1) ]
    all_labels =  [ i for i in range(1,k+1) ]

    ax.set_xticks(list(range(1,k+1)))
    ax.set_xticklabels(all_labels)

    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    plt.xlabel('signal')
    plt.ylabel('frequency')
    if len(dfc.index)==1:
        plt.suptitle('Frequency of Individual Signals', fontsize=13)
        plt.title(f'boutID={dfc.at[0,"boutID"]};\n'
                  f'BoutLen={len(dfc.at[0,"babbles"])}; '
                  f'Tr={dfc.at[0,"treatment"]}; '
                  #f'TrPe={dfc.at[0,"treatment_period"]}; '
                  #f'Age={dfc.at[0,"age"]}; '
                  f'Sex={dfc.at[0,"sex"]}',
                  fontsize=10)
    else:
        plt.title('Frequency of Individual Signals')

    fig.tight_layout()
    plt.savefig(basename + '_singles.png')

    return



def plot_pairs(data: dict, basename: str, dfc: pd.DataFrame):
 
    logging.info('plotting pairs')
    df = pd.DataFrame(columns=[i+1 for i in range(k)], index=[i+1 for i in range(k)], dtype=float)
    for data_col in data.keys():
        for data_row in data[data_col].keys():
            df[int(data_col[1:])][int(data_row[1:])] = float(data[data_col][data_row])

    # plot heatmap for pairs of sounds
    fig, ax = plt.subplots()
    im = ax.imshow(df, cmap='YlOrRd_r')
    odd_labels =  [ i if i%2==1 else ' ' for i in range(1,k+1) ]
    all_labels =  [ i for i in range(1,k+1) ]

    ax.set_xticks([i-1 for i in list(df.columns)])
    ax.set_yticks([i-1 for i in list(df.index)])
    ax.set_xticklabels(all_labels)
    ax.set_yticklabels(all_labels)

    #plt.setp(ax.get_xticklabels(), rotation=0, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    plt.xlabel('second signal in sequence')
    plt.ylabel('first signal in sequence')
    if len(dfc.index)==1:
        plt.suptitle('Frequency of Signal Pairs', fontsize=13)
        plt.title('test')
        plt.title(f'boutID={dfc.at[0,"boutID"]};\n'
                  f'BoutLen={len(dfc.at[0,"babbles"])}; '
                  f'Tr={dfc.at[0,"treatment"]}; '
                  #f'TrPe={dfc.at[0,"treatment_period"]}; '
                  #f'Age={dfc.at[0,"age"]}; '
                  f'Sex={dfc.at[0,"sex"]}',
                  fontsize=10)
    else:
        plt.title('Frequency of Signal Pairs')


    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('frequency', rotation=-90, va='bottom')

    fig.tight_layout()
    plt.savefig(basename + '_pairs.png')

    return()



def main():

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input csv file')
    parser.add_argument('-c', '--clean', action='store_true', required=False, default=False,
                        help='specify this flag if you are loading in cleaned data')
    parser.add_argument('-m', '--minlength', type=int, required=False, default=2,
                        help='minimum length for sequences to be used in pair analysis')
    parser.add_argument('-k', '--kmeans', type=int, required=False, default=39,
                        help='number of clusters from k means clustering')
    parser.add_argument('-a', '--analysis', type=str, required=False,
                        choices=['singles', 'pairs', 'triples', 'quads', 'quints', 'all'],
                        help='type of frequency analysis to perform')
    parser.add_argument('-d', '--dump', action='store_true', required=False,
                        help='specify this flag if you want to dump the sequences that go into the plot')
    parser.add_argument('-l', '--loglevel', type=str, required=False, default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='set log level')
    args = parser.parse_args()
    basename = args.input[:-4]
    global k
    k = args.kmeans
    
    # Set up logging
    format_str=f'[%(asctime)s] %(filename)s:%(funcName)s:%(lineno)s - %(levelname)s: %(message)s'
    logging.basicConfig(level=args.loglevel, format=format_str)

    # Extract
    logging.info('Reading in csv file')
    df = pd.read_csv(args.input)

    # Transform 
    logging.info('Transforming data')
    df_clean, basename = transform(df, args.clean, basename)

    if (args.dump == True):
        dump_bouts(df_clean, args.minlength, args.dump)

    # Analysis
    if (args.analysis == 'singles' or args.analysis == 'all'):
        analysis_singles(df_clean, args.minlength, basename)
    if (args.analysis == 'pairs' or args.analysis == 'all'):
        analysis_pairs(df_clean, args.minlength, basename)
    if (args.analysis == 'triples' or args.analysis == 'all'):
        analysis_triples(df_clean, args.minlength, basename)
    if (args.analysis == 'quads' or args.analysis == 'all'):
        analysis_quads(df_clean, args.minlength, basename)
    if (args.analysis == 'quints' or args.analysis == 'all'):
        analysis_quints(df_clean, args.minlength, basename)

    return



if __name__ == '__main__':
    main()

