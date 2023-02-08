import pandas as pd
import numpy as np
from scipy.stats import linregress


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with the SKU and price elasticity of logarithmic demand,
     where the elasticity is calculated as a coefficient of determination
    """

    df_copy = df.copy()

    df_copy['log_qty'] = np.log(df_copy['qty'] + 1)

    df_copy = df_copy.groupby('sku', as_index=False).apply(
        lambda x: pd.Series({'elasticity': linregress(x['price'], x['log_qty']).rvalue ** 2})
    )

    return df_copy
