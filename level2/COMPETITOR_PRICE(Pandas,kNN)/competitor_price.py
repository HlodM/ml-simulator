import pandas as pd
import numpy as np


def agg_comp_price(X: pd.DataFrame) -> pd.DataFrame:
    """Returns copy of input DataFrame grouped by sku with a 'new price' column that
    calculated by the aggregate function in the 'agg' column and competitors' prices
    according to the base price
    """

    x_copy = X.copy()

    # dictionary of aggregate functions
    func_dict = {
        'max': np.max,
        'min': np.min,
        'med': np.median,
        'avg': np.mean,
    }

    def aggregate_group_func(group):
        """Returns the  result of applying the aggregate function to the grouped Dataframe
        as a new comp_price column"""
        agg_func = group['agg'].iloc[0]

        if agg_func == 'rnk':
            return pd.Series({'comp_price': group['comp_price'].iloc[group['rank'].argmin()]})
        return pd.Series({'comp_price': func_dict[agg_func](group['comp_price'])})

    x_res = x_copy.groupby(['sku', 'agg', 'base_price'], as_index=False).apply(aggregate_group_func)

    # if the competitive and base prices differ by more than 20 %, the base price is taken,
    # otherwise - the competitive price
    x_res['new_price'] = np.where(
        abs(1 - x_res['comp_price'] / x_res['base_price']) <= 0.2,
        x_res['comp_price'],
        x_res['base_price']
    )

    return x_res
