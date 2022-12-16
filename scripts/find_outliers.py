import numpy as np
import pandas as pd
from tsmoothie.smoother import LowessSmoother

def get_limits(place):
        
    data = df.loc[df.placename == place].loc[:, ['doc_date', 'delta']]
    
    smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
    print(f'smoothing {place}')
    smoother.smooth((data.delta.astype(np.int64).values,))
    low, up = smoother.get_intervals('prediction_interval')
    
    return low, up


def align_outliers(place, low, up):
    
    data = df.loc[df.placename == place].loc[:, ['doc_date', 'delta']]
    
    aligned = pd.concat([data,
                         pd.Series(up[0], index=data.index, name='up'),
                         pd.Series(low[0], index=data.index, name='low')],
                         axis=1)
    
    aligned['outlier'] = ((aligned.delta > aligned.up) | (aligned.delta < aligned.low))
    
    return aligned


def calculate_outliers(df, places):
    
    outliers = pd.Series(data=False, index=df.index, name='outlier')
    
    for place in places:
        
        low, up = get_limits(place)
        aligned = align_outliers(place, low, up)
        outliers.loc[aligned.loc[aligned.outlier == True].index] = True
        
    return outliers


if __name__ == '__main__':

    df = pd.read_csv('../data/processed_data.tsv', sep='\t', encoding='utf8').convert_dtypes()
    df.doc_date = pd.to_datetime(df.doc_date)
    df.origin_date = pd.to_datetime(df.origin_date)

    outliers = calculate_outliers(df, df.placename.unique())
    outliers.to_csv('../data/outliers.tsv', sep='\t', encoding='utf8', index=False, )

    print('Finished!')
