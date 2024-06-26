import pandas as pd
import numpy as np

# config file
import configparser

import boto3

# DB connection
from sqlalchemy import create_engine, text  # database connection

# Feature engineering
from sklearn.preprocessing import MinMaxScaler

from time import sleep

from abc import ABC, abstractmethod

def sql_db_read(query: str, DB: str, driver:str = "mysql+pymysql", config_paths: str = "config/config.ini", dtype=None, index_col=None) -> pd.DataFrame:
    """
    Connects to a sql database and performs a query. Optionally, use non mysql driver.
    Args:
        query: SQL query
        DB: database to connect to
        driver: driver to use
        config_paths: path to config file
        dtype: Type name or dict of column -> type to coerce result DataFrame.
        index_col: Column(s) to set as index(MultiIndex).
    Returns:
        DataFrame object
    """
    config = configparser.ConfigParser()
    config.read(config_paths)

    user = config[DB]["user"]
    host = config[DB]["host"]
    port = int(config[DB]["port"])
    dbname = config[DB]["dbname"]
    password = config[DB]["password"]

    # Create the connection
    engine = create_engine(
        url="{0}://{1}:{2}@{3}:{4}/{5}".format(driver ,user, password, host, port, dbname))

    return pd.read_sql_query(sql=text(query), con=engine.connect(), index_col=index_col, dtype=dtype)


def snowflake_sql_db_read(query: str, DB: str, driver:str = "snowflake", config_paths: str = "config/config.ini", dtype=None, index_col=None) -> pd.DataFrame:
    """
    Connects to a Snowflake database and performs a query.
    Args:
        query: SQL query
        DB: database to connect to
        driver: driver to use
        config_paths: path to config file
        dtype: Type name or dict of column -> type to coerce result DataFrame.
        index_col: Column(s) to set as index(MultiIndex).
    Returns:
        DataFrame object
    """
    config = configparser.ConfigParser()
    config.read(config_paths)
    user = config[DB]["user"]
    password = config[DB]["password"]
    account = config[DB]["account"]
    warehouse = config[DB]["warehouse"]
    role = config[DB]["role"]
    dbname = config[DB]["dbname"]
    schema = config[DB]["schema"]
    # Create the connection
    engine = create_engine(
        url="{0}://{1}:{2}@{3}/{4}/{5}?warehouse={6}&role={7}".format(driver ,user, password, account, dbname, schema, warehouse, role))
    return pd.read_sql_query(sql=text(query), con=engine.connect(), index_col=index_col, dtype=dtype)


def aws_client(config_paths: str = "config/config.ini", service_name: str = "s3") -> boto3.client:
    """read the config file and return the aws client
    Args:
        config_paths (str): path to the config file
    Returns:
        client
    """
    config = configparser.ConfigParser()
    config.read(config_paths)

    aws_access_key_id = config["AWS"]["aws_access_key_id"]
    aws_secret_access_key = config["AWS"]["aws_secret_access_key"]

    return boto3.client(
        service_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,)




def categorical_encoding(df: pd.DataFrame, categorical_features: list = None, categorical_features_to_overweight: list = None, categorical_features_overweight_factor: float = 1) -> pd.DataFrame:
    """categorical encoding
    dummy variable encode and reweight according to number of unique values
    optional overweighting of certain categorical features
    Args:
        df (pandas.DataFrame): dataframe of bikes to feature engineer
        categorical_features (optional) (list): list of categorical features
        categorical_features_to_overweight (list) (optional):  list of categorical features to overweight
        categorical_features_overweight_factor (optional) (float): factor to overweight the categorical features by
    Returns:
        df (pandas.DataFrame): dataframe of bikes feature engineered
    """
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_features_to_overweight is None:
        categorical_features_to_overweight = []

    # Get the number of unique values for each categorical column
    unique_values_dict = {
        column: len(df[column].unique()) for column in categorical_features
    }

    # one hot encode the categorical features
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_features
    )

    # Adjust the weights of dummy variables according to the number of unique values in the original categorical column
    for encoded_column in df_encoded.columns:
        for original_column in unique_values_dict.keys():
            if original_column + '_' in encoded_column:
                # Get the number of unique values in the original categorical column from the dictionary
                num_unique_values = unique_values_dict[original_column]

                # Adjust the weight of the dummy variable
                df_encoded[encoded_column] = df_encoded[encoded_column] / \
                    np.sqrt(num_unique_values - 1)

                if original_column in categorical_features_to_overweight:
                    df_encoded[encoded_column] = df_encoded[encoded_column] * \
                        categorical_features_overweight_factor

    return df_encoded


def numerical_scaling(df: pd.DataFrame, categorical_features: list = None, numerical_features: list = None, numerical_features_to_overweight: list = None, numerical_features_overweight_factor: float = 1) -> pd.DataFrame:
    """numerical scaling for the bike dataframe
    minmax scaler and apply overweight
    Args:
        df (pandas.DataFrame): dataframe of bikes to feature engineer
        categorical_features (optional) (list): list of categorical features
        numerical_features (optional) (list): list of numerical features
        numerical_features_to_overweight (optional) (list): list of numerical features to overweight
        numerical_features_overweight_factor (optional) (float): factor to overweight the numerical features by
    Returns:
        df (pandas.DataFrame): dataframe of bikes feature engineered
    """
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    if numerical_features is None:
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()

    if numerical_features_to_overweight is None:
        numerical_features_to_overweight = []


    # scale the features
    df[numerical_features] = MinMaxScaler().fit_transform(df[numerical_features])

    # reweight the numerical features according to ratio to categorical features
    if len(numerical_features) > 0:
        df[numerical_features] = df[numerical_features] * \
            len(categorical_features)/len(numerical_features)
        # overweight certain numerical features
        df[numerical_features_to_overweight] = df[numerical_features_to_overweight] * \
            numerical_features_overweight_factor

    return df


def feature_engineering(
    df: pd.DataFrame,
    categorical_features: list,
    categorical_features_to_overweight: list,
    categorical_features_overweight_factor: float,
    numerical_features: list,
    numerical_features_to_overweight: list,
    numerical_features_overweight_factor: float,

) -> pd.DataFrame:
    """feature engineering for the bike dataframe,
    only keeps the categorical and numerical features
    one hot encodes the categorical features, devides by the square root of the unique number of unique categories
    min max scales the numerical features and reweights them according to the ratio of categorical to numerical features
    overweighing of categorical and numerical features available
    Args:
        df (pandas.DataFrame): dataframe of bikes to feature engineer
        categorical_features (list): list of categorical features
        categorical_features_to_overweight (list):
        categorical_features_overweight_factor (float):
        numerical_features (list): list of numerical features
        numerical_features_to_overweight (list)
        numerical_features_overweight_factor (float)

    Returns:
        df (pandas.DataFrame): dataframe of bikes feature engineered
    """

    df = df[categorical_features + numerical_features]
    df = categorical_encoding(df,
                              categorical_features,
                              categorical_features_to_overweight,
                              categorical_features_overweight_factor)
    df = numerical_scaling(df,
                           categorical_features,
                           numerical_features,
                           numerical_features_to_overweight,
                           numerical_features_overweight_factor)

    return df



# get the ids with a certain status
def get_data_status_mask(df: pd.DataFrame, status: list) -> pd.DataFrame:
    """get the ids with a certain status
    Args:
        df (pandas.DataFrame): dataframe of bikes
        status (list): list of status to filter by

    Returns:
        df (pandas.DataFrame): dataframe of bikes with the given status
    """

    mask = df.index[df["status"].isin(status)].tolist()

    return mask

def get_numeric_frame_size(frame_size_code, bike_type_id, default_value=56):
    """Convert frame_size_code and bike_type_id to a numeric value.
    Args:
        frame_size_code (str or int): frame size code to convert
        bike_type_id (int): bike type identifier
        default_value (int): default value if conversion is not possible
    Returns:
        int: numeric value of frame size code
    """
    # Mapping dictionary
    frame_size_code_to_cm = {
        1: {
            "xxxs": 43,
            "xxs": 46,
            "xs": 49,
            "s": 52,
            "m": 54,
            "l": 57,
            "xl": 60,
            "xxl": 62,
            "xxxl": 64,
        },
        2: {
            "xxxs": 30,
            "xxs": 33,
            "xs": 36,
            "s": 41,
            "m": 46,
            "l": 52,
            "xl": 56,
            "xxl": 58,
            "xxxl": 61,
        },
    }

    # If frame_size_code is already numeric, return it as is
    if isinstance(frame_size_code, (np.float64, np.int64)):
        return int(frame_size_code)

    # Convert frame_size_code to string in case it's not and try to get the numeric value
    frame_size_code = str(frame_size_code).lower()
    return frame_size_code_to_cm.get(bike_type_id, {}).get(frame_size_code, default_value)

def frame_size_code_to_numeric(df: pd.DataFrame, bike_type_id_column="bike_type_id", frame_size_code_column="frame_size_code", default_value=56) -> pd.DataFrame:
    """Map string frame_size_code with numeric frame_size_code and assign a default value if missing or not in mapping.
    Args:
        df (pandas.DataFrame): dataframe of bikes
        bike_type_id_column (str): column name of bike_type_id
        frame_size_code_column (str): column name of frame_size_code
        default_value (int or str): default value to assign if frame_size_code is missing or not in mapping
    Returns:
        df (pandas.DataFrame): dataframe of bikes with numeric frame_size_code
    """
    # Apply the get_numeric_frame_size function to each row
    df[frame_size_code_column] = df.apply(
        lambda row: get_numeric_frame_size(row[frame_size_code_column], row[bike_type_id_column], default_value), axis=1
    )
    # Ensure the frame_size_code column is of type int
    df[frame_size_code_column] = df[frame_size_code_column].astype(int)
    return df

class DataStoreBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def get_logging_info(self):
        pass

    def read_data_periodically(self, period, logger):
        """Read data periodically
        Args:
            period: period in minutes
            logger: logger
        """
        error = None
        period = period * 60
        while True:
            try:
                self.read_data()
                log_info = self.get_logging_info()
                if log_info:
                    logger.info("Data read", extra=log_info)
                sleep(period)
            except Exception as error:
                logger.error("Data could not be read: " + str(error))
                sleep(period)

