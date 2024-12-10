import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import argparse
import logging
import ast

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to preprocess the data
def preprocess_data(data):
    logging.info('Preprocessing data...')

    # Map 'Sex' and 'Treatment' columns to numeric values
    data['Sex'] = data['Sex'].map({'M': 0, 'F': 1})
    data['Treatment'] = data['Treatment'].map({'CONTROL': 0, 'CORT': 1, 'OIL': 2})

    # Convert strings in 'Babbles' column to lists
    data['Babbles'] = data['Babbles'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Convert date columns to datetime
    date_columns = ['Hatch date', 'Fledge date', 'Date on vocalization']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    # Extract statistics from 'Babbles' column
    def process_babbles(babble_list):
        return {
            'babble_count': len(babble_list),
            'babble_mean': sum(babble_list) / len(babble_list) if babble_list else 0,
            'babble_sum': sum(babble_list),
        }

    babbles_stats = data['Babbles'].apply(process_babbles)
    data['Babble Length'] = babbles_stats.apply(lambda x: x['babble_count'])
    data['Babble Mean'] = babbles_stats.apply(lambda x: x['babble_mean'])
    data['Babble Sum'] = babbles_stats.apply(lambda x: x['babble_sum'])

    logging.info('Data preprocessing completed.')
    return data

# Function to compute ANOVA
def compute_anova(data, factors, response):
    # Ensure the response is the last column in the formula
    formula = f"{response} ~ " + " * ".join(factors)
    logging.info(f"ANOVA Formula: {formula}")
    
    # Fit the model
    model = ols(formula, data=data).fit()
    
    # Compute ANOVA
    anova_result = anova_lm(model)
    logging.info("\nANOVA Results:")
    logging.info(anova_result)
    
    # Check if p-value is significant
    significant = anova_result["PR(>F)"].min() < 0.05
    logging.info(f"\nSignificant P-Value Found: {'Yes' if significant else 'No'}")
    return anova_result

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Perform ANOVA on user-selected columns.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("-f", "--factors", nargs='+', required=True, help="List of independent variables separated by space.")
    args = parser.parse_args()

    # Hardcoded dependent variable (response)
    response = "Babble_Length"

    # Load the dataset or use the default
    logging.info('Reading and preparing data for analysis')
    if args.input:
        data = pd.read_csv(args.input)
    else:
        logging.warning("No data file provided.")

    # Preprocess the data
    data = preprocess_data(data)

    print(args.factors)

    # Ensure column names are standardized, ANOVA cnt read spaces
    data.columns = data.columns.str.replace(' ', '_')
    args.factors = [factor.replace(' ', '_') for factor in args.factors]

    # Extract selected columns
    selected_columns = args.factors + [response]
    try:
        selected_data = data[selected_columns]
    except KeyError as e:
        logging.warning(f"Error: {e}")
        logging.warning("Ensure the selected columns exist in the DataFrame or CSV file.")
        return

    logging.info(f"Columns selected for ANOVA: {selected_columns}")
    
    # Compute ANOVA
    compute_anova(selected_data, args.factors, response)

if __name__ == "__main__":
    main()
