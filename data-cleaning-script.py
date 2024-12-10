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
    
    logging.info('Renaming columns')
    df = df.rename(columns=config["rename_columns"])
    
    # logging.info('Casting columns to specified data types')
    # for column, dtype in config["data_types"].items():
    #     df[column] = df[column].astype(dtype)
    
    logging.info('Starting to format previously cleaned data')
    logging.info('Grouping data by Bout ID and aggregating Babbles into lists')
    df = df.groupby("Bout ID", as_index=False).agg({
        **{col: "first" for col in df.columns if col not in ["Bout ID", "Babbles"]},
        "Babbles": list
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
    logging.info('Starting to dump individual bouts (Babbles)')
    bout_list = df_clean[df_clean['Babbles'].str.len() >= minlength]
    count = len(bout_list)

    if dump:
        for i, row in bout_list.iterrows():
            logging.info('Dumping a bout')
            row.to_csv(f'dump_{i+1}_clean.csv', encoding='utf-8', index=False)

    logging.info(f'{count} bouts were greater than or equal to {minlength} signals')
    return count


def analysis_singles(df_clean: pd.DataFrame, minlength: int, basename: str):
    singles = count_singles(df_clean, minlength)

    with open(basename + '_singles.json', 'w') as o:
        json.dump(singles, o, indent=2)
    with open(basename + '_singles.csv', 'w') as o:
        writer = csv.writer(o)
        writer.writerow(['signal', 'freq'])
        for item in singles.keys():
            writer.writerow([item, singles[item]])
    
    logging.info('Single sequence analysis complete')
    plot_singles(singles, basename, df_clean)

    return


def analysis_pairs(df_clean: pd.DataFrame, minlength: int, basename: str):
    # Analyze and save pairs signal sequences that meet minimum length requirement.
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
    
    logging.info('Pair sequence analysis complete')
    plot_pairs(pairs, basename, df_clean)

    return


def analysis_triples(df_clean: pd.DataFrame, minlength: int, basename: str):
    if minlength < 3:
        minlength = 3
        logging.info('Setting minlength to 3 for triples analysis')

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
    logging.info('Triple sequence analysis complete')
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
        logging.info('Setting minlength to 4 for quads analysis')

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
    """
    Count the frequency of single signals (sequences of 1 elements) in a DataFrame.

    Returns:
    dict: A dictionary where the keys are the single signal IDs and the values are their counts.
    """
    logging.info('Starting to count singles')
    signals = [sig for row in df_clean[df_clean['Babbles'].str.len() >= minlength]['Babbles'] for sig in row]
    freq_singles = dict.fromkeys(range(1, k+1), 0)
    freq_singles.update(dict(Counter(signals)))
    logging.info('Finished counting singles')

    return (freq_singles)


def count_pairs(df_clean: pd.DataFrame, minlength: int) -> dict:
    """
    Count the frequency of pairs signals (sequences of 2 elements) in a DataFrame.

    Returns:
    dict: A dictionary where the keys are the pairs signal IDs and the values are their counts.
    """
    logging.info('Starting to count pairs')
    
    # Initialize the pairs dictionary with zeros
    freq_pairs = {f'a{i}': {f'b{j}': 0 for j in range(1, k + 1)} for i in range(1, k + 1)}
    
    # Filter sequences by length and count pairs
    valid_sequences = df_clean[df_clean['Babbles'].str.len() >= minlength]
    counter = len(valid_sequences)
    
    for sequence in valid_sequences['Babbles']:
        # Count pairs using zip
        for first, second in zip(sequence, sequence[1:]):
            if f'a{first}' in freq_pairs and f'b{second}' in freq_pairs[f'a{first}']:
                freq_pairs[f'a{first}'][f'b{second}'] += 1
            else:
                logging.debug(f'Invalid pair found: {first}, {second}')

    logging.info(f'Processed {counter} Bouts that are >= {minlength} signals')
    logging.info('Finished counting pairs')
    
    return (freq_pairs)


def count_triples(df_clean: pd.DataFrame, minlength: int) -> dict:
    """
    Count the frequency of triple signals (sequences of 3 elements) in a DataFrame.

    Returns:
    dict: A dictionary where the keys are the triple signal IDs and the values are their counts.
    """
    logging.info('Starting to count triples')
    
    # Initialize the triples dictionary with zeros
    freq_triples = { f'a{i}': {f'b{j}': {f'c{k}': 0 for k in range(1, k + 1)} for j in range(1, k + 1)} for i in range(1, k + 1)}

    # Filter sequences by length and count triples
    valid_sequences = df_clean[df_clean['Babbles'].str.len() >= minlength]
    counter = len(valid_sequences)

    for sequence in valid_sequences['Babbles']:
        # Count triples using zip
        for first, second, third in zip(sequence, sequence[1:], sequence[2:]):
            key_a = f'a{first}'
            key_b = f'b{second}'
            key_c = f'c{third}'
            if key_a in freq_triples and key_b in freq_triples[key_a] and key_c in freq_triples[key_a][key_b]:
                freq_triples[key_a][key_b][key_c] += 1
            else:
                logging.debug(f'Invalid triple found: {first}, {second}, {third}')

    logging.info(f'Processed {counter} Bouts that are >= {minlength} signals')
    logging.info('Finished counting triples')

    return (freq_triples)


def count_quads(df_clean: pd.DataFrame, minlength: int) -> dict:
    logging.info('Starting to count quads')

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
        if ( len(row['Babbles']) < minlength ):
            continue
        else:
            counter += 1
            for first, second, third, fourth in zip(row['Babbles'], row['Babbles'][1:], row['Babbles'][2:], row['Babbles'][3:]):
                try:
                    freq_quads['a'+str(first)]['b'+str(second)]['c'+str(third)]['d'+str(fourth)] += 1
                except KeyError as e:
                    logging.debug(f'KeyError for {first} or {second} or {third} or {fourth}')

    logging.info(f'Processed {counter} Bouts that are >= {minlength} signals')
    logging.info('Finished counting quads')

    return(freq_quads)


def count_quints(df_clean: pd.DataFrame, minlength: int) -> dict:
    logging.info('Starting to count quints')

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
        if ( len(row['Babbles']) < minlength ):
            continue
        else:
            counter += 1
            for first, second, third, fourth, fifth in zip(row['Babbles'], row['Babbles'][1:], row['Babbles'][2:], row['Babbles'][3:], row['Babbles'][4:]):
                try:
                    freq_quints['a'+str(first)]['b'+str(second)]['c'+str(third)]['d'+str(fourth)]['e'+str(fifth)] += 1
                except KeyError as e:
                    logging.debug(f'KeyError for {first} or {second} or {third} or {fourth} or {fifth}')

    logging.info(f'Processed {counter} Bouts that are >= {minlength} signals')
    logging.info('Finished counting quints')

    return(freq_quints)


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
        plt.title(f"boutID={dfc.at[0,'Bout ID']}; BoutLen={len(dfc.at[0,'Babbles'])}; Tr={dfc.at[0,'treatment']}; Sex={dfc.at[0,'sex']}", fontsize=10)
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
            df.loc[row, col] = int(data[a_sig][b_sig])

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
        plt.title(f'boutID={row["Bout ID"]}; BoutLen={len(row["Babbles"])}; '
                    f'Tr={row["treatment"]}; Sex={row["sex"]}', fontsize=10)
    else:
        plt.title('Frequency of Signal Pairs')

    # Add colorbar and save
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Frequency', rotation=-90, va='bottom')
    
    fig.tight_layout()
    plt.savefig(basename + '_pairs.png')

    return()


def setup_model_data(df_clean: pd.DataFrame, columns: list, basename: str):
    """
    Prepares data for a machine learning model by creating separate CSV files
    for each user-specified column, including required columns (Babbles, Bout ID) in each file.
    
    Parameters:
    df_clean (pandas.DataFrame): The input data.
    columns (list): A list of column names to include.
    basename (str): A base name for new CSV files.

    Returns:
    pandas.DataFrame: Ordered DataFrame with Babbles, Bout ID, and specified columns.
    list: Paths to created CSV files
    """
    # Set up column order: required columns first, then user columns
    required_cols = ['Babbles', 'Bout ID']
    
    # Validation
    if not columns:
        raise ValueError("Please provide at least one column.")
    
    missing_cols = [col for col in columns if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {', '.join(missing_cols)}")

    # Create individual CSV files for each user-specified column
    created_files = []
    for col in columns:
        # Create DataFrame with required columns plus the current column
        col_data = df_clean[required_cols + [col]].copy()
        
        # Generate filename based on column name (sanitize the column name)
        safe_colname = col.replace(' ', '_')
        col_csv_path = f"{basename}_{safe_colname}_scm.csv"
        
        # Export to CSV
        col_data.to_csv(col_csv_path, encoding='utf-8', index=False)
        created_files.append(col_csv_path)
        logging.info(f'Data exported for column {col}')
    
    return 


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
    parser.add_argument('-sc', '--sequenceclass', type=lambda s: [item.strip() for item in s.split(',')], required=False, 
                        help='Provide at least one column name, separated by commas, to configure the data input for the model, ex: "column1, column2, column3"')
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
    if (args.analysis == 'triples' or args.analysis == 'all'):
        analysis_triples(df_clean, args.minlength, basename)
    if (args.analysis == 'quads' or args.analysis == 'all'):
        analysis_quads(df_clean, args.minlength, basename)
    if (args.analysis == 'quints' or args.analysis == 'all'):
        analysis_quints(df_clean, args.minlength, basename)


    # Sequence Classification Model Set Up
    logging.info('Configuring the data input for the Sequence Classification Model')
    columns = args.sequenceclass
    setup_model_data(df_clean, columns, basename)

    return

if __name__ == "__main__":
    main()