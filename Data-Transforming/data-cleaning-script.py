import pandas as pd
import json
import logging

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

def clean_and_transform_data(config_file):
    """
    Cleans and transforms the input data file based on the provided configuration.
    Parameters:
    config_file (str): Path to the JSON configuration file.
    Returns:
    pandas.DataFrame: The cleaned and transformed DataFrame.

    """
    logging.info('Starting data transformation process')
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    logging.info('Loading data from CSV')
    df = pd.read_csv(config["data_file"], low_memory=False)
    
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
    
    logging.info('Grouping data by Bout ID and aggregating Cluster6 into lists')
    transformed_df = df.groupby("Bout ID (sans subtype)", as_index=False).agg({
        **{col: "first" for col in df.columns if col not in ["Bout ID (sans subtype)", "Cluster6"]},
        "Cluster6": list
    })

    logging.info('Starting to dump individual bouts (Cluster6)')
    # Step 1: Calculate the minimum length of lists in "Cluster6"
    min_length = transformed_df["Cluster6"].apply(len).min()

    # Step 2: Filter out rows where the length of "Cluster6" is less than the min length
    df = transformed_df[transformed_df["Cluster6"].apply(len) >= min_length]
    logging.info(f'{(df.shape)} bouts were greater than or equal to {min_length} signals')


    logging.info('Data transformation completed')
    return df

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig( level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

    # Transform     
    cleaned_data = clean_and_transform_data("./Data-Transforming/clean_data.json")
        
    # Save final formatted data
    logging.info('Saving formatted data to CSV')
    cleaned_data.to_csv("./Data-Transforming/cleaned_data.csv", index=False)

    logging.info('Process completed successfully')