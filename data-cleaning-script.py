import argparse
import csv
import json
import logging
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

k=6

# def apply_column_transformation(df, column, transformation_name):
#     """
#     Applies the specified transformation to the given column in the DataFrame.
#     Parameters:
#     df (pandas.DataFrame): The input DataFrame.
#     column (str): The name of the column to transform.
#     transformation_name (str): The name of the transformation to apply.
#     Returns:
#     pandas.DataFrame: The DataFrame with the column transformed.
#     """

#     logging.info(f'Applying {transformation_name} transformation to column {column}')
#     transformations = {
#         "strip": lambda x: x.str.strip(),
#         "to_datetime": lambda x: pd.to_datetime(x)
#     }
#     transformation = transformations.get(transformation_name, None)
#     if transformation:
#         df[column] = transformation(df[column])
#     return df

def clean_and_transform_data(config_file: str, basename: str):
    """
    Cleans and transforms the input data file based on the provided configuration.
    Parameters:
    config_file (str): Path to the JSON configuration file.
    Returns:
    pandas.DataFrame: The cleaned and transformed DataFrame.

    """

    with open(config_file, "r") as f:
        config = json.load(f)
    
    logging.info('Loading data from CSV')
    df = pd.read_csv(config["data_file"], low_memory=False)
    
    logging.info('Starting to clean data')
    logging.info('Selecting specified columns')
    df = df[config["columns"]]
    
    # logging.info('Applying custom transformations')
    # for column, transformation in config["transformations"].items():
    #     df = apply_column_transformation(df, column, transformation)
    
    # logging.info('Filling missing values')
    # df = df.fillna(config["fill_na"])
    
    # logging.info('Renaming columns')
    # df = df.rename(columns=config["rename_columns"])
    
    # logging.info('Casting columns to specified data types')
    # for column, dtype in config["data_types"].items():
    #     df[column] = df[column].astype(dtype)
    
    logging.info('Starting to format previously cleaned data')
    logging.info('Grouping data by Bout ID and aggregating Cluster6 into lists')
    df = df.groupby("Bout ID (sans subtype)", as_index=False).agg({
        **{col: "first" for col in df.columns if col not in ["Bout ID (sans subtype)", "Cluster6"]},
        "Cluster6": list
    })

    logging.info('Writing clean and transformed data to csv file')
    df.to_csv(basename + '_clean.csv', encoding='utf-8', index=False)
    
    return(df)



def dump_bouts(df_clean: pd.DataFrame, minlength: int, dump: bool):
    """
    Dump individual bouts from a DataFrame that meet a minimum length requirement.

    Parameters:
    df_clean (pd.DataFrame): The input DataFrame.
    minlength (int): The minimum length requirement for a bout to be dumped.
    dump (bool): Whether to actually dump the bouts or not.

    Returns:
    int: The number of bouts that were greater than or equal to the minimum length.
    """
    logging.info('Starting to dump individual bouts (Cluster6)')
    bout_list = df_clean[df_clean['Cluster6'].str.len() >= minlength]
    count = len(bout_list)

    if dump:
        for i, row in bout_list.iterrows():
            logging.info('Dumping a bout')
            row.to_csv(f'dump_{i+1}_clean.csv', encoding='utf-8', index=False)

    logging.info(f'{count} bouts were greater than or equal to {minlength} signals')
    return count


# Analysis 1 - 5
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



def count_singles(df_clean: pd.DataFrame, minlength: int) -> dict:
    """
    Count the frequency of single signals that meet a minimum length requirement.

    Parameters:
    df_clean (pd.DataFrame): The input DataFrame.
    minlength (int): The minimum length requirement for a single signal to be counted.

    Returns:
    dict: A dictionary where the keys are the single signal IDs and the values are their counts.
    """
    logging.info('Starting to count singles')
    signals = [sig for row in df_clean[df_clean['Cluster6'].str.len() >= minlength]['Cluster6'] for sig in row]
    freq_singles = dict.fromkeys(range(1, k+1), 0)
    freq_singles.update(dict(Counter(signals)))
    logging.info('Finished counting singles')

    return freq_singles



def count_pairs(df_clean: pd.DataFrame, minlength: int) -> dict:
    logging.info('Starting to count pairs')

    freq_pairs = {}
    for vala in [ 'a'+str(i) for i in range(1,k+1) ]:
        freq_pairs[vala] = {}
        for valb in [ 'b'+str(i) for i in range(1,k+1) ]:
            freq_pairs[vala][valb] = 0

    counter=0
    for index, row in df_clean.iterrows():
        if ( len(row['Cluster6']) < minlength ):
            continue
        else:
            counter += 1
            for first, second in zip(row['Cluster6'], row['Cluster6'][1:]):
                try:
                    freq_pairs['a'+str(first)]['b'+str(second)] += 1
                except KeyError as e:
                    logging.debug(f'KeyError for {first} or {second}')

    logging.info(f'{counter} bouts were greater than or equal to {minlength} signals')
    logging.info('Finished counting pairs')

    return(freq_pairs)



def plot_singles(data: dict, basename: str, dfc: pd.DataFrame):
    """
    Plot a histogram of the frequency of single signals.

    """
    logging.info('Plotting singles')

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(list(range(1, k+1)), list(data.values()), color='indianred')

    # Set the x-axis ticks and labels
    ax.set_xticks(list(range(1, k+1)))
    ax.set_xticklabels([i if i % 2 == 1 else '' for i in range(1, k+1)], fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    # Add labels and title
    plt.xlabel('Signal', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    if len(dfc.index) == 1:
        plt.suptitle('Frequency of Individual Signals', fontsize=14)
        plt.title(f"boutID={dfc.at[0,'Bout ID (sans subtype)']}; BoutLen={len(dfc.at[0,'Cluster6'])}; Tr={dfc.at[0,'treatment']}; Sex={dfc.at[0,'sex']}", fontsize=10)
    else:
        plt.title('Frequency of Individual Signals', fontsize=14)

    fig.tight_layout()
    plt.savefig(basename + '_singles.png')

    return()


def plot_pairs(data: dict, basename: str, dfc: pd.DataFrame):
   """
   Plot heatmap of signal pair frequencies.

   """
   logging.info('Plotting pairs')

   # Create and populate frequency matrix 
   df = pd.DataFrame(0, columns=range(1,k+1), index=range(1,k+1), dtype=float)
   for a_sig in data:
       for b_sig in data[a_sig]:
           row, col = int(b_sig[1:]), int(a_sig[1:])
           df.loc[row, col] = float(data[a_sig][b_sig])

   # Create plot
   fig, ax = plt.subplots(figsize=(10,8))
   im = ax.imshow(df, cmap='YlOrRd_r')

   # Configure axes
   ax.set_xticks(range(k))
   ax.set_yticks(range(k)) 
   ax.set_xticklabels(range(1,k+1), fontsize=10)
   ax.set_yticklabels(range(1,k+1), fontsize=10)
   ax.set_xlabel('Second Signal in Sequence')
   ax.set_ylabel('First Signal in Sequence')

   # Add title
   if len(dfc.index) == 1:
       row = dfc.iloc[0]
       plt.suptitle('Frequency of Signal Pairs', fontsize=13)
       plt.title(f'boutID={row["Bout ID (sans subtype)"]}; BoutLen={len(row["Cluster6"])}; '
                f'Tr={row["treatment"]}; Sex={row["sex"]}',
                fontsize=10)
   else:
       plt.title('Frequency of Signal Pairs')

   # Add colorbar and save
   cbar = fig.colorbar(im)
   cbar.ax.set_ylabel('Frequency', rotation=-90, va='bottom')
   
   fig.tight_layout()
   plt.savefig(basename + '_pairs.png')

   return()


def main():

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='input json file to read steps to clean and trasnfrom the csv file')
    # parser.add_argument('-c', '--clean', action='store_true', required=False, default=False,
    #                     help='specify this flag if you are loading in cleaned data')
    parser.add_argument('-m', '--minlength', type=int, required=False, default=2,
                        help='minimum length for sequences to be used in pair analysis')
    parser.add_argument('-k', '--kmeans', type=int, required=False, default=6,
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
    
    basename = 'CMBabble_Master'
    config_file = args.input
    global k
    k = args.kmeans

    # Configure logging
    logging.basicConfig( level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

    # Extract
    logging.info('Reading in json file')
    config_file = args.input

    # Clean and Transform  
    df_clean = clean_and_transform_data(config_file, basename)
        
    logging.info('Cleaning and Transforming data process completed successfully')

    if (args.dump == True):
        dump_bouts(df_clean, args.minlength, args.dump)


    # Analysis
    if (args.analysis == 'singles' or args.analysis == 'all'):
        analysis_singles(df_clean, args.minlength, basename)
    if (args.analysis == 'pairs' or args.analysis == 'all'):
        analysis_pairs(df_clean, args.minlength, basename)
    # if (args.analysis == 'triples' or args.analysis == 'all'):
    #     analysis_triples(df_clean, args.minlength, basename)
    # if (args.analysis == 'quads' or args.analysis == 'all'):
    #     analysis_quads(df_clean, args.minlength, basename)
    # if (args.analysis == 'quints' or args.analysis == 'all'):
    #     analysis_quints(df_clean, args.minlength, basename)

    return

if __name__ == "__main__":
    main()