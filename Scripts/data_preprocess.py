import pandas as pd
import numpy as np
from scipy.stats import zscore

class DataPreprocessor:
    def __init__(self, df):
        """
        Initialize the preprocessor with a copy of the DataFrame.
        """
        self.df = df.copy()

    def handle_nulls(self, numeric_method='mean', categorical_method='mode', drop=False):
        """
        Handles null values in the DataFrame.
        
        Parameters:
          - numeric_method: For numeric columns, choose one of 'mean', 'median', or 'mode'
                            to fill missing values. (Ignored if drop=True.)
          - categorical_method: For string columns, 'mode' is used by default, or you can
                                supply a default value.
          - drop: If True, drop rows with any null values.
        """
        if drop:
            self.df = self.df.dropna()
        else:
            # Handle numeric columns
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    if numeric_method == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif numeric_method == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif numeric_method == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    else:
                        # Fallback to dropping null values if method is unrecognized
                        self.df = self.df[self.df[col].notnull()]
            
            # Handle categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    if categorical_method == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    else:
                        self.df[col].fillna(categorical_method, inplace=True)

    def handle_nulls_specific_columns(self, columns, numeric_method='mean', categorical_method='mode'):
        """
        Handles null values for a specified list of columns.
        
        Parameters:
          - columns: list of column names to handle null values for.
          - numeric_method: For numeric columns, choose one of 'mean', 'median', 'mode',
                            or 'drop' to drop rows with nulls in that column.
          - categorical_method: For string columns, choose 'mode' to fill with mode, 
                                supply a default value, or 'drop' to drop rows with nulls.
        """
        for col in columns:
            if col not in self.df.columns:
                print(f"Column '{col}' does not exist in the DataFrame.")
                continue
            if self.df[col].isnull().sum() > 0:
                # Process numeric columns
                if self.df[col].dtype in ['int64', 'float64']:
                    if numeric_method == 'drop':
                        self.df = self.df[self.df[col].notnull()]
                    elif numeric_method == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif numeric_method == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif numeric_method == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    else:
                        print(f"Unrecognized numeric method for column '{col}': {numeric_method}")
                # Process categorical columns
                elif self.df[col].dtype == object:
                    if categorical_method == 'drop':
                        self.df = self.df[self.df[col].notnull()]
                    elif categorical_method == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    else:
                        # Use provided default value for categorical columns
                        self.df[col].fillna(categorical_method, inplace=True)

    def handle_outliers(self):
        """
        Detects and removes rows with outliers using a democratic approach.
        A value is considered an outlier in a numeric column if at least two out of three methods
        (IQR, Z-score, MAD) flag it.
        """
        # Identify numeric columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        outlier_indices = set()

        for col in numerical_cols:
            # Use non-null data for calculations
            col_data = self.df[col].dropna()

            # IQR Method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = set(col_data[(col_data < lower_bound) | (col_data > upper_bound)].index)

            # Z-score Method
            z_scores = zscore(col_data)
            z_outliers = set(col_data[np.abs(z_scores) > 3].index)

            # MAD Method
            median = np.median(col_data)
            mad = np.median(np.abs(col_data - median))
            mad_threshold = 3 * mad
            mad_outliers = set(col_data[np.abs(col_data - median) > mad_threshold].index)

            # Democratic approach: flag index if flagged by at least 2 methods
            combined_indices = set.union(iqr_outliers, z_outliers, mad_outliers)
            for idx in combined_indices:
                count = 0
                if idx in iqr_outliers:
                    count += 1
                if idx in z_outliers:
                    count += 1
                if idx in mad_outliers:
                    count += 1
                if count >= 2:
                    outlier_indices.add(idx)

        print(f"Removing {len(outlier_indices)} rows flagged as outliers across numeric columns.")
        self.df = self.df.drop(index=outlier_indices)
        return self.df

    def handle_duplicates(self):
        """
        Finds and drops duplicate rows. Before dropping duplicates, prints out one example
        of a duplicate record (with all of its duplicate occurrences).
        """
        duplicates = self.df[self.df.duplicated(keep=False)]
        if not duplicates.empty:
            # Pick the first duplicate row and print all occurrences of that record.
            first_dup_index = duplicates.index[0]
            duplicate_record = self.df.loc[first_dup_index]
            # Create a mask for all rows identical to the duplicate_record.
            mask = (self.df == duplicate_record).all(axis=1)
            dup_group = self.df[mask]
            print("Example duplicate group found:")
            print(dup_group)
        else:
            print("No duplicates found.")

        # Drop duplicate rows.
        self.df = self.df.drop_duplicates()

    def drop_columns(self, columns):
        """
        Drops the specified columns from the DataFrame.
        
        Parameters:
          - columns: list of column names to be dropped.
        """
        self.df = self.df.drop(columns=columns)
        print(f"Dropped columns: {columns}")

    def get_dataframe(self):
        """
        Returns the processed DataFrame.
        """
        return self.df

