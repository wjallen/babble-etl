import pandas as pd
import ast
import gc

from itertools import combinations
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from multiprocessing import Pool, cpu_count


# Function to clean and prepare the entire dataset or a chunk
def clean_and_prepare_data(chunk):
    # Convert date columns to datetime (vectorized)
    date_columns = ['Hatch date', 'Fledge date', 'Date on vocalization']
    for col in date_columns:
        if col in chunk.columns:
            chunk[col] = pd.to_datetime(chunk[col], errors='coerce')

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
        'No._eggs_hatched_from_nest': 'Number_eggs_hatched_from_nest',
        'No._birds_fledged_from_nest': 'Number_birds_fledged_from_nest'
    })

    return chunk


# Function to precompute header combinations
def get_header_combinations(csv_file, exclude_headers=[]):
    df = pd.read_csv(csv_file, nrows=0)  # Only reads headers
    headers = df.columns.str.replace(' ', '_').tolist()

    filtered_headers = [header for header in headers if header not in exclude_headers]

    # Precompute all header combinations
    all_combinations = [
        comb for r in range(1, len(filtered_headers) + 1) 
        for comb in combinations(filtered_headers, r)
    ]
    return all_combinations


# Function to run ANOVA in parallel
def run_anova_parallel(args):
    chunk, combination, response_col = args
    try:
        data = chunk[list(combination) + [response_col]].dropna()
        if data.empty:
            return None

        # Construct formula and run ANOVA
        formula = f'{response_col} ~ ' + ' * '.join(combination)
        model = ols(formula, data=data).fit()
        anova_result = anova_lm(model)
        anova_result['Combination'] = str(combination)
        return anova_result
    except Exception as e:
        print(f"Error with combination {combination}: {e}")
        return None


# Process CSV in chunks
def process_csv(csv_file, header_combinations, response_col='Babble_Length', chunksize=50000):
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
    pd.concat(results).to_csv('partial_anova_results.csv', index=False)


# Main block
if __name__ == "__main__":
    csv_file = "CMBabble_Master_clean.csv"  # Replace with your file path
    exclude_headers = ["Babbles", "Bout_ID", "Notes", "Raven work", "Date_on_vocalization_2", ""]
    response_col = 'Babble_Length'

    # Precompute header combinations
    header_combinations = get_header_combinations(csv_file, exclude_headers)

    # Process the file and run ANOVA
    process_csv(csv_file, header_combinations, response_col)

    # Filter significant results
    filter_significant_results(file='partial_anova_results.csv', output_file='filtered_file.csv')
