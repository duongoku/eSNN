class DataFrameMapper:
    def __init__(self, features, df_out=False):
        self.features = features
        self.df_out = df_out

    def fit_transform(self, X):
        import pandas as pd
        import numpy as np

        result_columns = []
        result_transformers = []

        for columns, transformer in self.features:

            if isinstance(columns, str):
                columns = [columns]

            input_df = X[columns]
            if transformer is not None:
                transformed = transformer.fit_transform(input_df)
                if len(transformed.shape) == 1:
                    transformed = transformed.reshape(-1, 1)
            else:
                transformed = input_df.values

            result_transformers.append(transformed)

            if hasattr(transformer, "get_feature_names_out"):
                new_cols = transformer.get_feature_names_out()
            else:
                # Generate column names based on the actual number of output columns
                if transformed.shape[1] > len(columns):
                    new_cols = [
                        f"{col}_{i}"
                        for col in columns
                        for i in range(transformed.shape[1] // len(columns))
                    ]
                else:
                    new_cols = columns

            if isinstance(new_cols, str):
                new_cols = [new_cols]
            result_columns.extend(new_cols)

        final_result = np.hstack(result_transformers)

        if self.df_out:
            return pd.DataFrame(final_result, columns=result_columns, index=X.index)
        return final_result
