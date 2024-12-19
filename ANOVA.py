import pandas as pd
import ast
import gc
import logging

from itertools import combinations
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from multiprocessing import Pool, cpu_count

# Initialize logging
logging.basicConfig(level=logging.INFO)

def clean_and_prepare_data(chunk):
    """
    Retrieve data from csv file to clean and prepare the entire dataset
    :param chunk: The dataset
    :return: The dataset that are clean  and prepare

    TODO: Fix the date columns to datetime, ANOVA can only read in certain formate
    """
    logging.info('Starting to retrieve data to clean and prepare the entire dataset')


    # Convert date columns to datetime (vectorized)
    # date_columns = ['Hatch date', 'Fledge date', 'Date on vocalization']
    # for col in date_columns:
    #     if col in chunk.columns:
    #         chunk[col] = pd.to_datetime(chunk[col], errors='coerce')

    # Replace spaces with underscores in column names
    chunk.columns = chunk.columns.str.replace(' ', '_')

    # Extract statistics from 'Babbles' column (optimized with vectorized processing)
    if 'Babbles' in chunk.columns:
        def process_babbles_vectorized(babbles):
            try:
                babble_list = ast.literal_eval(babbles)
                if isinstance(babble_list, list):
                    return len(babble_list), sum(babble_list) / len(babble_list), sum(babble_list)
                else:
                    return 0, 0, 0
            except (ValueError, SyntaxError):
                return 0, 0, 0

        babble_stats = chunk['Babbles'].apply(process_babbles_vectorized)
        chunk[['Babble_Length', 'Babble_Mean', 'Babble_Sum']] = pd.DataFrame(babble_stats.tolist(), index=chunk.index)

    # Rename columns
    chunk = chunk.rename(columns={
        'Bout_no.': 'Bout_number',
    })

    logging.info('Finished cleaning and preparing the entire dataset\n')
    return chunk


def get_header_combinations(csv_file, exclude_headers=[]):
    """
    Retrieve data from csv file to extract headers that will used and some to exclued
    :param csv_file: The path to the csv_file
    :param exclude_headers: A list of headers to remove if needed
    :return: A list of combinations to preform ANOVA Testing
    """
    logging.info('Starting to extract headers that will used and some to exclued')
    df = pd.read_csv(csv_file, nrows=0) 
    headers = df.columns.str.replace(' ', '_').tolist()

    filtered_headers = [header for header in headers if header not in exclude_headers]
    logging.info('Finsihed extracting headers')

    # Precompute all header combinations
    all_combinations = [
        comb for r in range(1, len(filtered_headers) + 1) 
        for comb in combinations(filtered_headers, r)
    ]
    logging.info('Finsihed finding all combinations for ANOVA Testing\n')
    return all_combinations


def run_anova_parallel(args):
    """
    Run the ANOVA in parallel
    Retrieves info need to format the ANOVA Formula to run Testing
    :param args: A list of headers to remove if needed
    :return: The Anova Result from testing 
    """
    chunk, combination, response_col = args
    try:
        data = chunk[list(combination) + [response_col]].dropna()
        if data.empty:
            return None

        # Construct formula and run ANOVA
        formula = f'{response_col} ~ ' + ' * '.join(combination)
        model = ols(formula, data=data).fit()
        anova_result = anova_lm(model)

        # Add meaningful names for effects (Interaction effect between 2+ variables )
        anova_result['Interaction Effects'] = anova_result.index
        anova_result['Combination'] = str(combination)
        
        return anova_result
    except Exception as e:
        print(f"Error with combination {combination}: {e}")
        return None


def process_csv(csv_file, header_combinations, response_col='Babble_Length', chunksize=50000):
    """ 
    Retrieves all vars needed to get results from run_anova_parallel to Process CSV in chunks
    :param csv_file: Path to CVS File
    :param header_combinations: A list of combinations of headers 
    :param response_col: Response Col for testing (Dependent Var)
    :param chunksize: Size of datasets being 
    :return: CSV file of the Anova Result
    """
    logging.info('Starting running combinations for ANOVA Testing')
    results = []
    chunk_iter = pd.read_csv(csv_file, chunksize=chunksize)

    for chunk in chunk_iter:
        chunk = clean_and_prepare_data(chunk)
        # Prepare arguments for parallel processing
        args = [(chunk, combo, response_col) for combo in header_combinations]

        # Use multiprocessing for ANOVA
        with Pool(cpu_count()) as pool:
            partial_results = pool.map(run_anova_parallel, args)
            results.extend([res for res in partial_results if res is not None])

        gc.collect()

    # Save all results at once
    logging.info('Saving ANOVA results to partial_anova_results.csv\n')
    pd.concat(results).to_csv('partial_anova_results.csv', index=False)


def filter_significant_results(file='partial_anova_results.csv', output_file='filtered_file.csv'):
    """
    Filters rows where PR(>F) is <= 0.05, removes duplicates based on 'Interaction Effects',
    and keeps the one with the lowest PR(>F).
    :param file: Path to CSV file.
    :param output_file: Path to output CSV file.
    :return: Filtered and deduplicated CSV file.
    """
    logging.info('Starting to filter rows where PR(>F) is less than or equal to 0.05')
    df = pd.read_csv(file)

    # Filter rows where PR(>F) is not NaN and <= 0.05
    df_filtered = df[df['PR(>F)'].notna() & (df['PR(>F)'] <= 0.05)]
    
    # Sort by 'Interaction Effects' and 'PR(>F)' to then Remove duplicates based on 'Interaction Effects', keeping the one with the lowest PR(>F)
    df_filtered = df_filtered.sort_values(by=['Interaction Effects', 'PR(>F)'])
    df_filtered = df_filtered.drop_duplicates(subset=['Interaction Effects'], keep='first')

    # Sort the final results by PR(>F) in ascending order
    df_filtered = df_filtered.sort_values(by='PR(>F)', ascending=True).reset_index(drop=True)

    df_filtered.to_csv(output_file, index=False)
    logging.info(f"Significant and deduplicated results saved to '{output_file}'")


if __name__ == "__main__":
    # csv_file = "CMBabble_Master_clean.csv" 
    csv_file = "CMBabble_Master_combined.csv" 
    exclude_headers = ["Babbles", "Bout_ID", "Notes", "Raven work", "Date_on_vocalization_2"]
    response_col = 'Babble_Length'

    # Precompute header combinations
    header_combinations = get_header_combinations(csv_file, exclude_headers)

    # Process the file and run ANOVA
    process_csv(csv_file, header_combinations, response_col)

    # Filter significant results
    filter_significant_results(file='partial_anova_results.csv', output_file='filtered_file.csv')
