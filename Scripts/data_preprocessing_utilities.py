import pandas as pd

def read_csv(path):
    return pd.read_csv(path)

def print_head(df):
    print(df.head())

def print_dtypes(df):
    print(df.dtypes)

def print_null_count(df):
    print(df.isnull().sum())

def print_shape(df):
    print("No of rows =", df.shape[0], ", No of columns =", df.shape[1])

def summarize_data(df, columns=None, stats=['mean', 'median', 'std']):
    """
    Prints mean, median, and standard deviation for the given columns.
    
    Parameters:
    df (pd.DataFrame): The dataframe to analyze.
    columns (list, optional): List of column names to summarize. Uses all if None.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns  # Use all numeric columns if none provided
    
    summary = {}

    # Add stats to summary dynamically
    if 'min' in stats:
        summary['min'] = df[columns].min()
    if 'max' in stats:
        summary['max'] = df[columns].max()
    if 'range' in stats:
        summary['range'] = df[columns].max() - df[columns].min()
    if 'mean' in stats:
        summary['mean'] = df[columns].mean()
    if 'median' in stats:
        summary['median'] = df[columns].median()
    if 'std' in stats:
        summary['std'] = df[columns].std()
    if 'IQR' in stats:
        summary['IQR'] = df[columns].quantile(0.75) - df[columns].quantile(0.25)
    
    # Display the results
    print("*********** Dataset summary of numerical columns ***********")
    print(pd.DataFrame(summary))

def handle_nulls_by_column(df, column, mode):
    if mode == 'delete':
        return df[~df[column].isnull()]
    elif mode == 'drop':
        df.drop(column, axis=1, inplace=True)
        return df
    else:
        print('invalid mode of handling nulls, enter correct mode')

def convert_column_to_datetime(df, column):
    return pd.to_datetime(df[column])