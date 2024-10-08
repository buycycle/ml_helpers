import pandas as pd
import numpy as np

# config file
import configparser


# DB connection
from sqlalchemy import create_engine, text  # database connection

# Feature engineering
from sklearn.preprocessing import MinMaxScaler

from time import sleep

from abc import ABC, abstractmethod
import json


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

    with engine.connect() as connection:
        return pd.read_sql_query(sql=text(query), con=connection, index_col=index_col, dtype=dtype)


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
        mask (list): bikes indicies with the given status
    """

    mask = df.index[df["status"].isin(status)].tolist()

    return mask

def get_preference_mask(df: pd.DataFrame, preference: dict) -> pd.DataFrame:
    """Get the indices that match a certain preference dict.
    Args:
        df (pandas.DataFrame): dataframe of items
        preference (dict): preferences dict with feature as key and value(s) to filter as value, also lists are allowed
    Returns:
        mask (list): indices of items with the given preferences
    """
    # Start with all indices
    mask = df.index.tolist()
    # Narrow down the indices based on each preference
    for feature, value in preference.items():
        if feature in df.columns:
            if isinstance(value, list):  # Check if the value is a list (multiple values)
                # Use isin to filter by multiple values and update the mask
                mask = df.index[df[feature].isin(value)].intersection(mask).tolist()
            else:
                # Single value, proceed as before
                mask = df.index[df[feature] == value].intersection(mask).tolist()
    return mask

def get_preference_mask_condition(df: pd.DataFrame, preferences: tuple) -> list:
    """Get the indices that match a certain preference tuple.
    Args:
        df (pandas.DataFrame): dataframe of items
        preferences (tuple): preferences tuple with feature as key and a lambda function to filter as value
    Returns:
        mask (list): indices of items with the given preferences
    """
    # Start with a mask that includes all items
    mask = pd.Series([True] * len(df), index=df.index)

    # Narrow down the mask based on each preference
    for feature, condition in preferences:
        if feature in df.columns:
            # Apply the lambda function to get the condition mask
            condition_mask = condition(df)
            # Combine with the existing mask using logical AND
            mask &= condition_mask
    # Convert the boolean mask to a list of indices
    return mask[mask].index.tolist()

def get_preference_mask_condition_list(df: pd.DataFrame, preferences: tuple) -> list:
    """
    Returns a list of indices indicating items in df that match any of the combined conditions
    specified by the preferences tuple. Each combined condition corresponds to a full set of
    preferences for a single user.
    Args:
        df (pd.DataFrame): The DataFrame containing items to be filtered.
        preferences (tuple): A tuple where each item is a pair consisting of a
            feature (column name) and a list of lambda functions. Each lambda
            function represents a combined filtering condition for that feature.
    Returns:
        list: A list of indices where True indicates that the item at the
            corresponding index in the DataFrame fulfills the combined filtering conditions.
    """
    mask = pd.Series([False] * len(df), index=df.index)
    for feature, conditions in preferences:
        feature_mask = pd.Series([False] * len(df), index=df.index)
        for condition in conditions:
            condition_mask = condition(df)
            feature_mask |= condition_mask
        mask |= feature_mask
    return mask.index[mask].tolist()

def get_numeric_frame_size(frame_size_code, bike_type_id=1, default_value=56):
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
            "none": 1,
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

    # Attempt to convert frame_size_code to an integer if it's a string that represents a number
    try:
        return int(frame_size_code)
    except ValueError:
        # If conversion fails, it's not a simple number string; proceed with the mapping
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

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.
    Converts NumPy integers to Python ints, NumPy floats to Python floats,
    and NumPy arrays to lists. Other types are handled by the superclass.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

