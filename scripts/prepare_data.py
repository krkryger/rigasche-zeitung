import sys
import pandas as pd
import re
import convertdate
import json
from tqdm import tqdm

datapath = sys.argv[1]

print('Importing raw data')
main_df = pd.read_parquet(datapath)

print('Loading auxiliaries')
with open('re_pattern.txt', 'r', encoding='utf8') as f:
    pattern = re.compile(f.read())

with open('re_exceptions.txt', 'r', encoding='utf8') as f:
    exceptions = []
    for line in f.readlines():
        exceptions.append(line.strip('\n'))

with open('placename_replacement_dict.json', 'r', encoding='utf8') as f:
    placename_replacement_dict = json.load(f)


def scan_placenames_dates(main_df, exceptions):
    
    results = []
    
    for ix, row in tqdm(main_df.iterrows()):
        matches = re.finditer(pattern, row.full_text)
        for m in matches:
            results.append([ix, row.date] + list(m.groupdict().values()) + [m.span()[0], m.span()[1]])
        
    result_df = pd.DataFrame(columns=['doc_id', 'doc_date', 'placename', 'day', 'day2', 'month', 'start', 'end'],
                        data=results).fillna(pd.NA)
    
    result_df['doc_date'] = pd.to_datetime(result_df['doc_date'], format='%Y-%m-%d')
    result_df['year'] = result_df['doc_date'].dt.year
    
    result_df['day'] = result_df['day'].astype(int)
    result_df['day'] = result_df['day'].apply(lambda x: x if (x in range(1,32)) else pd.NA)
    
    result_df['day2'] = result_df['day2'].str.lstrip('( ').str.strip('.) ')
    result_df['day2'].fillna(0, inplace=True)
    result_df['day2'] = result_df['day2'].astype(int)
    result_df.replace(0, pd.NA, inplace=True)

    return result_df[~result_df.placename.isin(exceptions)]


def cleanup_dates(df):
    
    #df = df.copy()
    
    month_dict = dict(zip(['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun',
                           'Jul', 'Aug', 'Sept', 'Okt', 'Nov', 'Dec'], range(1,13)))
    
    df.month = df.month.str.capitalize()
    df.month.replace({'Dez': 'Dec', 'Mar': 'Mär', 'Oct': 'Okt', '0ct': 'Okt', '0kt': 'Okt',
                      'Jnl': 'Jul', 'Jnn': 'Jun', 'Juu': 'Jun', 'May': 'Mai'}, inplace=True)
    
    df.month.replace(month_dict, inplace=True)
    df.loc[df.month.isin(['C', 'C.', 'D. m', 'D. m.']), 'month'] = df.loc[df.month.isin(['C', 'C.', 'D. m', 'D. m.']), 'doc_date'].dt.month
    df.loc[df.month.isin(['V. m', 'V. m.']), 'month'] = (df.loc[df.month.isin(['v. M', 'V. m.']), 'doc_date'] - pd.DateOffset(months=1)).dt.month - 1
    
    df.month = df.month.fillna(0).astype(int)
    df.month = df.month.apply(lambda x: x if (x in range(1,13)) else pd.NA)
    
    df.day = df.day.fillna(0).astype(int)
    df.day = df.day.apply(lambda x: x if (x in range(1,32)) else pd.NA)
    
    df.day2 = df.day2.fillna(0).astype(int)
    df.day2 = df.day2.apply(lambda x: x if (x in range(1,32)) else pd.NA)
    
    return df   


def verify_dates(ix, df):
    
    #print(df.loc[ix,['day', 'day2', 'month', 'doc_date']])
    
    day, day2, month, doc_date = df.loc[ix,['day', 'day2', 'month', 'doc_date']]
    
    if type(day) == pd._libs.missing.NAType or type(month) == pd._libs.missing.NAType:
        return pd.NA
    
    # 2 kp
    if type(day2) == int:
        #print('difference: ', max(day, day2) - min(day, day2))
        if month == doc_date.month or month < doc_date.month:
            if day > day2 and day > doc_date.day:
                # julian is the smaller one
                origin_date = pd.to_datetime(f'{str(doc_date.year)}-{month}-{day2}',
                                             format='%Y-%m-%d', errors='coerce')
            else: # kontrolli see üle
                origin_date = pd.to_datetime(f'{str(doc_date.year)}-{month}-{day}',
                                             format='%Y-%m-%d', errors='coerce')
                
        elif month > doc_date.month:
            if month == 12 and doc_date.month == 1:
                if day > day2 and day > doc_date.day:
                    origin_date = pd.to_datetime(f'{str(doc_date.year-1)}-{month}-{day2}',
                                                 format='%Y-%m-%d', errors='coerce')
                else:
                    origin_date = pd.to_datetime(f'{str(doc_date.year-1)}-{month}-{day}',
                                             format='%Y-%m-%d', errors='coerce')
                    
            else:
                return pd.NA
                
            
    
    # 1 kp
    else:
        # mis on kuu võrreldes lehe kuuga?
        if month < doc_date.month:
            # origin is julian, apply directly to datetime
            origin_date = pd.to_datetime(f'{str(doc_date.year)}-{month}-{day}',
                                         format='%Y-%m-%d', errors='coerce')
        
        elif month == doc_date.month:
            if day > doc_date.day:
                # origin is gregorian, convert to julian first, then apply
                try:
                    day_jul = convertdate.julian.from_gregorian(year=doc_date.year, month=month, day=day)
                except ValueError:
                    return pd.NA
                origin_date = pd.to_datetime(f'{str(day_jul[0])}-{str(day_jul[1])}-{str(day_jul[2])}',
                                             format='%Y-%m-%d', errors='coerce')
            else:
                # origin is julian, apply directly to datetime
                origin_date = pd.to_datetime(f'{str(doc_date.year)}-{month}-{day}',
                                             format='%Y-%m-%d', errors='coerce')
        
        elif month > doc_date.month:
            if month == 12 and doc_date.month == 1:
                # origin is from last year - probably in julian b/c is smaller that doc_date and there is no day2
                origin_date = pd.to_datetime(f'{str(doc_date.year-1)}-{month}-{day}',
                                             format='%Y-%m-%d', errors='coerce')
            else:
                # origin is from next month, thus in gregorian
                try:
                    day_jul = convertdate.julian.from_gregorian(year=doc_date.year, month=month, day=day)
                except ValueError:
                    return pd.NA
                origin_date = pd.to_datetime(f'{str(day_jul[0])}-{str(day_jul[1])}-{str(day_jul[2])}',
                                             format='%Y-%m-%d', errors='coerce')
            
    return origin_date




print('Scanning raw data for placenames and dates')
df = scan_placenames_dates(main_df, exceptions)

df.placename.replace(placename_replacement_dict, inplace=True)

print('Formatting date info')
df = cleanup_dates(df)

print('Calculating origin dates and calendar differences')
origin_dates = []
for ix in tqdm(df.index):
    origin_dates.append(verify_dates(ix, df))

df['origin_date'] = origin_dates
df.dropna(subset=['origin_date'], inplace=True) # drop entries with no valid date found
df['origin_date'] = pd.to_datetime(df.origin_date)

df['delta'] = (df['doc_date'] - df['origin_date']).dt.days

# drop entries with origin date that is negative or more than 1 year
df = df.loc[(df.delta > 0) & (df.delta < 350)] 

print('Saving the dataframe')
df.to_csv('processed_data.tsv', sep='\t', encoding='utf8', index=False)

print('Finished')

