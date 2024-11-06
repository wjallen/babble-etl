import pandas as pd
import json
import logging

# Configure logging
logging.basicConfig( level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def apply_column_transformation(df, column, transformation_name):
    """
    Applies the specified transformation to the given column in the DataFrame.
    """
    logging.info(f'Applying {transformation_name} transformation to column {column}')
    transformations = {
        "strip": lambda x: x.str.strip(),
        "to_datetime": lambda x: pd.to_datetime(x)
    }
    transformation = transformations.get(transformation_name, None)
    if transformation:
        df[column] = transformation(df[column])
    return df

def clean_and_transform_data(config_file):
    """
    Cleans and transforms the input data file based on the provided configuration.
    """
    logging.info('Starting data transformation process')
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    logging.info('Loading data from CSV')
    df = pd.read_csv(config["data_file"])
    
    logging.info('Selecting specified columns')
    df = df[config["columns"]]
    
    logging.info('Applying custom transformations')
    for column, transformation in config["transformations"].items():
        df = apply_column_transformation(df, column, transformation)
    
    logging.info('Filling missing values')
    df = df.fillna(config["fill_na"])
    
    logging.info('Renaming columns')
    df = df.rename(columns=config["rename_columns"])
    
    logging.info('Casting columns to specified data types')
    for column, dtype in config["data_types"].items():
        df[column] = df[column].astype(dtype)
    
    logging.info('Data transformation completed')
    return df

if __name__ == "__main__":
    cleaned_data = clean_and_transform_data("/Data-Cleaning/clean_data.json")
    logging.info('Saving cleaned data to CSV')
    cleaned_data.to_csv("path/to/cleaned_data.csv", index=False)
    logging.info('Process completed successfully')