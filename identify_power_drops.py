import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Read CSV file into a DataFrame and preprocess it
df = pd.read_csv('power_calcs_data_05_20_2024.csv')
df = df.drop(columns=(['group_id',
                      'launch_id',
                      'stack_id',
                      'stack_position_bottom_to_top',
                      'launch_t_nav',
                      'days_since_launch',
                      'long_period_amplitude',]))

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.sort_values(by=['identifier', 'string', 'date'], inplace=True)

# Disregard measurements before 2021
df = df[df['date'] >= '2021-01-01']


# identify all power drops for a specific string of a satellite's array
def find_drops(df, sat, string):

    # Smoothing the function with a moving average
    window_size = 5
    df['smoothed_power'] = df['power'].rolling(window=window_size).mean()

    # Remove zero values
    df = df[df['smoothed_power'] != 0]
    df = df[np.isnan(df['smoothed_power']) == False]

    # Calculate the derivative
    df['derivative'] = df['smoothed_power'].diff() / df['date'].diff().dt.days

    # Calculate the difference over a rolling window
    window_size_diff = 5  # Adjust as needed
    df['smoothed_power_diff'] = df['smoothed_power'].diff(periods=window_size_diff)

    # Identify big changes based on the derivative
    threshold_diff = -100 # magnitude difference between points should be large
    threshold_derivative = -10  # Derivative should be large for a sharp drop
    big_changes = df[(df['derivative'] <= threshold_derivative) & (df['smoothed_power_diff'] <= threshold_diff)]

    # add information of identified drops to power_drops.csv
    big_changes.to_csv('power_drops.csv', mode='a', index=False, header=False)


    # Plot the smoothed function and drop points
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['smoothed_power'], label='Smoothed Power')
    plt.scatter(big_changes['date'], big_changes['smoothed_power'], color='red', label='Big Changes')
    plt.xlabel('Date')
    plt.ylabel('Power')
    plt.title(f'Smoothed Power with large changes: Sat {sat}: String {string}')
    plt.legend()
    plt.grid(True)

    if not os.path.exists(f'drop_plots/sat_{sat}'):
        os.mkdir(f'drop_plots/sat_{sat}')
    plt.savefig(f'drop_plots/sat_{sat}/str_{string}')


# identify all drops for every string for each satellite
id_list = set(df['identifier'])
string_list = set(df['string'])

for id in id_list:
    for string in string_list:
        filt_df = df[(df['identifier'] == id) & (df['string'] == string)]
        find_drops(filt_df, id, string)
