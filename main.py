import pandas as pd

# Load the sample data
df = pd.read_csv("/mnt/data/leave_sample.csv")

# Create proper DATE column
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# Ensure proper sorting
df = df.sort_values(by=['EMPLOYEE_ID', 'DATE'])

# Get the next day's TYPE for the same worker
df['NEXT_TYPE'] = df.groupby('EMPLOYEE_ID')['TYPE'].shift(-1)

# Filter for the specific transitions of interest
patterns_of_interest = [
    ('UNK', 'RDO'),     # sick leave → RDO
    ('UNK', 'PER_L'),   # sick leave → personal leave
    ('RDO', 'UNK'),     # RDO → sick leave
    ('PER_L', 'UNK')    # personal leave → sick leave
]

pattern_df = df[df[['TYPE', 'NEXT_TYPE']].apply(tuple, axis=1).isin(patterns_of_interest)]

# Count occurrences of each pattern by DEPO_NAME
pattern_counts = pattern_df.groupby(['DEPO_NAME', 'TYPE', 'NEXT_TYPE']).size().reset_index(name='COUNT')

