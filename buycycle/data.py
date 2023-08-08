import pandas as pd
import numpy as np

# DB connection
from sqlalchemy import create_engine, text  # database connection

# Feature engineering
from sklearn.preprocessing import MinMaxScaler



def frame_size_code_to_numeric(df: pd.DataFrame, bike_type_id_column="bike_type_id", frame_size_code_column="frame_size_code") -> pd.DataFrame:
    """map string frame_size_code with numeric frame_size_code
    Args:
        df (pandas.DataFrame): dataframe of bikes
        bike_type_id_column (str): column name of bike_type_id
        frame_size_code_column (str): column name of frame_size_code

    Returns:
        df (pandas.DataFrame): dataframe of bikes with numeric frame_size_code
    """
    # for each bike_type_id replace the frame_size_code with a numiric value from a dictionery
    frame_size_code_to_cm = {
        1:  {
            "xxs": "46",
            "xs": "49",
            "s": "52",
            "m": "54",
            "l": "57",
            "xl": "60",
            "xxl": "62",
            "xxxl": "64",
        },
        2:  {
            "xxs": "33",
            "xs": "36",
            "s": "41",
            "m": "46",
            "l": "52",
            "xl": "56",
            "xxl": "58",
            "xxxl": "61",
        },
    }

    # Filter dataframe to only include rows where frame_size_code is in the dictionary for the given bike_type_id and is non-numeric
    mask = (
        df[frame_size_code_column].isin(
            ["xxs", "xs", "s", "m", "l", "xl", "xxl", "xxxl"]
        )
        & df[frame_size_code_column].str.isnumeric().eq(False)
    )

    # Replace the frame_size_code with the numeric value from the dictionary
    df.loc[mask, frame_size_code_column] = df.loc[mask].apply(
        lambda row: frame_size_code_to_cm[row[bike_type_id_column]][row[frame_size_code_column]], axis=1
    )

    # Transform the frame_size_code to numeric, for the already numeric but in string format
    df[frame_size_code_column] = pd.to_numeric(df[frame_size_code_column])

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


def categorical_encoding(df: pd.DataFrame, categorical_features: list, categorical_features_to_overweight: list, categorical_features_overweight_factor: float) -> pd.DataFrame:
    """categorical encoding for the bike dataframe
    dummy variable encode and reweight according to number of unique values
    Args:
        df (pandas.DataFrame): dataframe of bikes to feature engineer
        categorical_features (list): list of categorical features
        categorical_features_to_overweight (list): list of categorical features to overweight
        categorical_features_overweight_factor (float): factor to overweight the categorical features by
    Returns:
        df (pandas.DataFrame): dataframe of bikes feature engineered
    """

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


def numerical_scaling(df: pd.DataFrame, categorical_features: list, numerical_features: list, numerical_features_to_overweight: list, numerical_features_overweight_factor: int) -> pd.DataFrame:
    """numerical scaling for the bike dataframe
    minmax scaler and apply overweight
    Args:
        df (pandas.DataFrame): dataframe of bikes to feature engineer
        categorical_features (list): list of categorical features
        numerical_features (list): list of numerical features
        numerical_features_to_overweight (list): list of numerical features to overweight
        numerical_features_overweight_factor (int): factor to overweight the numerical features by
    Returns:
        df (pandas.DataFrame): dataframe of bikes feature engineered
    """
    # scale the features
    df[numerical_features] = MinMaxScaler().fit_transform(df[numerical_features])
    # reweight the numerical features according to ratio to categorical features
    df[numerical_features] = df[numerical_features] * \
        len(categorical_features)/len(numerical_features)
    # overweight certain numerical features
    df[numerical_features_to_overweight] = df[numerical_features_to_overweight] * \
        numerical_features_overweight_factor
    return df


