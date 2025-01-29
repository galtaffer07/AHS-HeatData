import os
import pandas as pd
import matplotlib as plt
import folium
from geopy.geocoders import Bing
from folium.plugins import HeatMap
import tqdm
import sqlalchemy
import numpy as np
import datetime
from meteostat import Hourly, Daily, Point

#LIST OF DATASETS
#r"C:\Users\galta\Downloads\AHS_2019_CO2.csv"
#r"C:\Users\galta\Downloads\2023 Q3Q4 AHS ZN-T & CO2 UTC.csv"
#r"C:\Users\galta\Downloads\2023 Q1Q2 AHS ZN-T & CO2 UTC.csv"
#print(f"Pandas version: {pd.__version__}")

temperatures = pd.read_csv(r"C:\Users\galta\Downloads\AHS_2019_Temps.csv", skipfooter = 3, engine = 'python')

carbon_levels = pd.read_csv(r"C:\Users\galta\Downloads\2023 Q3Q4 AHS ZN-T & CO2 UTC.csv", skipfooter = 3, engine = 'python', index_col=0) 
pd.set_option('display.max_rows', 10, 'display.max_columns', None)
#print(carbon_levels)
carbon_levels_without_empty_columns = carbon_levels[[c for c in carbon_levels.columns if not c.startswith('Unnamed: ')]]  #removes the 175 empty columns at end of dataset
pd.DataFrame.iteritems = pd.DataFrame.items
pd.set_option("display.max.columns", None)
#carbon_levels.dropna(axis = 1, inplace = True)
carbon_levels_without_empty_columns.dropna(axis = 0)



#PROCESS FOR FILTERING
Holidays_for_script = [
    datetime.date(2024, 1, 1),
    datetime.date(2024, 1, 15),
    datetime.date(2024, 2, 19),
    datetime.date(2024, 2, 20),
    datetime.date(2024, 2, 21),
    datetime.date(2024, 2, 22),
    datetime.date(2024, 2, 23),
    datetime.date(2024, 4, 15),
    datetime.date(2024, 4, 16),
    datetime.date(2024, 4, 17),
    datetime.date(2024, 4, 18),
    datetime.date(2024, 4, 19),
    datetime.date(2024, 5, 27),
    datetime.date(2024, 6, 19)
    #datetime.date(2019, 2, 18), #when the data starts - feb break
    #datetime.date(2019, 2, 19),
    #datetime.date(2019, 2, 20),
    #datetime.date(2019, 2, 21),
    #datetime.date(2019, 2, 22),
    #datetime.date(2019, 4, 15), #april break
    #datetime.date(2019, 4, 16),
    #datetime.date(2019, 4, 17),
    #datetime.date(2019, 4, 18),
    #datetime.date(2019, 4, 19),
    #datetime.date(2019, 5, 27) #memorial day
] 

early_release_days = [
    datetime.date(2024, 1, 26),
    datetime.date(2024, 3, 8),
    datetime.date(2024, 5, 3)
    #datetime.date(2019, 2, 1),
    #datetime.date(2019, 3, 15),
    #datetime.date(2019, 5, 3)
]
SCHOOL_START_TIME = datetime.time(8, 15) #7:45 for old data, 8:15 for new
NORMAL_RELEASE_TIME = datetime.time(2, 51) #2:20 for old data, 2:51 for new
EARLY_RELEASE_TIME = datetime.time(11, 30) #10:50 for old data, 11:30 for new

carbon_levels_without_empty_columns.index = pd.to_datetime(carbon_levels_without_empty_columns.index)
#print(carbon_levels.index)
#print(carbon_levels.index.time)

time_is_not_notschool = ~np.isin(carbon_levels_without_empty_columns.index.time, (pd.date_range("8:00", "15:00", freq = "1min").time))
carbon_levels_school = carbon_levels_without_empty_columns[~time_is_not_notschool] #filters out of school hours


time_is_not_weekend = ~np.isin(carbon_levels_school.index.weekday, [5,6]) #filters weekends
carbon_levels_not_weekends = carbon_levels_school[time_is_not_weekend]

#print(carbon_levels_not_weekends)

time_is_not_holiday = ~np.isin(carbon_levels_not_weekends.index.date, Holidays_for_script) #filters holidays
carbon_levels_without_holidays = carbon_levels_not_weekends[time_is_not_holiday]

is_early_release_day = np.isin(carbon_levels_without_holidays.index.date, early_release_days) #filters early release days
is_within_early_release_hours = (
    is_early_release_day
    & (carbon_levels_without_holidays.index.time >= SCHOOL_START_TIME)
    & (carbon_levels_without_holidays.index.time <= EARLY_RELEASE_TIME)
)
is_within_normal_school_hours = (
    ~is_early_release_day
    & (carbon_levels_without_holidays.index.time >= SCHOOL_START_TIME)
    & (carbon_levels_without_holidays.index.time <= NORMAL_RELEASE_TIME)
)
carbon_levels_in_normal_school_hours = carbon_levels_without_holidays[
    (is_within_early_release_hours | is_within_normal_school_hours)
]



week = carbon_levels_without_holidays.index.to_period('W-MON')
month = carbon_levels_without_holidays.index.to_period('M')
grouped_by_week = carbon_levels_without_holidays.groupby(week) #makes a dataset that groups times by week
grouped_by_month = carbon_levels_without_holidays.groupby(month) #makes a dataset that groups all times by month

#FINDING RANGE OF DATA TO USE
sorted_column = carbon_levels_without_holidays['Cafe UV08 ZN08 CO2'].sort_values()
first_benchmark = sorted_column.quantile(0.25)
last_benchmark = sorted_column.quantile(0.75)
data_range = last_benchmark - first_benchmark
data_within_range = carbon_levels_without_holidays[(carbon_levels_without_holidays['Cafe UV08 ZN08 CO2'] >= first_benchmark) & (carbon_levels_without_holidays['Cafe UV08 ZN08 CO2'] <= last_benchmark)]

#print(str(carbon_levels_without_holidays['CC RTU06 ZN-T'].quantile(0.25)) + "-" + str(carbon_levels_without_holidays['CC RTU06 ZN-T'].quantile(0.75)))


check_base = carbon_levels.drop(columns = carbon_levels.columns[0], axis = 1)


#for column_name, column_content in carbon_levels_without_holidays.iteritems():          # used to detect if sensor is broken + if sensor is detecting values of over 800 ppm
#    for row_name, row_content in carbon_levels_without_holidays.iterrows():
#       if (((carbon_levels_without_holidays.at[row_name, column_name]) > 800) | (((carbon_levels_without_holidays.at[row_name, column_name]) < 400) & ((carbon_levels_without_holidays.at[row_name, column_name]) > 100))):
#           if((carbon_levels_without_holidays.at[row_name, column_name]) == 2000):
#                print(str(column_name) + " is broken (value is 2000)")
#           else:
#               print(str(row_name) + ' ' + str(column_name) + " recorded a level of " + str(carbon_levels_without_holidays.at[row_name, column_name]) + " ppm.")

#print("Weekly values: " + str(grouped_by_week['RM309 ZN14 Q CO2'].mean())) 
#print(grouped_by_month['RM309 ZN14 Q CO2'].mean()) #prints out monthly averages
'''
v = "input"
terminate_loop = False
targ1 = 70
targ2 = 80   #please change as needed

def flag_sensors(df, min_level=1600, max_level=1900):
    flagged_sensors = []
    for sensor in df.columns:
        if df[sensor].between(min_level, max_level).any():
            flagged_sensors.append(sensor)
    return flagged_sensors

def sort_sensors(df, min_level=1600, max_level=1900):
    high = []
    medium = []
    light = []

    for sensor in df.columns:
        count = df[sensor].between(min_level, max_level).sum()
        if count >= 50:
            high.append(sensor)
        elif 5 <= count < 25:
            medium.append(sensor)
        elif count > 0:
            light.append(sensor)

    return high, medium, light

def broken_sensors(df, min_level=1900, max_level=5000):
    broken_sensors = []
    for sensor in df.columns:
        if df[sensor].between(min_level, max_level).any():
            broken_sensors.append(sensor)
        return broken_sensors

def find_rooms_needing_AC(df):
    sensor_medians={}

    for sensor in df.columns:
        top_temps = df[sensor].nlargest(21)
        sensor_medians[sensor] = top_temps.median()

    sorted_sensors = sorted(sensor_medians.items(), key=lambda x: x[1], reverse = True)
    return sorted_sensors

temperature = [col for col in carbon_levels_without_holidays.columns if 'CO2' not in col.upper()]#checks to see if columns in dataset do not have CO2 in name => temperature sensor
print(find_rooms_needing_AC(carbon_levels_without_holidays[temperature]))'''

latitude = 42.6572
longitude = -71.1555
location_point = Point(latitude, longitude)
'''
def fetch_weather(dt):
    #use hourly data to match specific datetime
    start = dt
    end = dt
    weather = Hourly(location_point, start, end).fetch()

    if not weather.empty:
        row = weather.loc[weather.index[0]] #first row matching datetime
        return row['temp'], row['prcp'] #Fetch temp and precipitation

    return None, None

if 'temperature' not in carbon_levels_without_holidays.columns or 'precipitation' not in carbon_levels_without_holidays.columns:
    carbon_levels_without_holidays[['temperature', 'precipitation']] = carbon_levels_without_holidays.index.to_series().apply(lambda dt: pd.Series(fetch_weather(dt)))
'''

temp_columns=[col for col in carbon_levels_without_holidays.columns if "ZN-T" in col]


#weather_temp_column = 'temperature'
inside_threshold = int(input("Enter a threshold value to check: "))
carbon_levels_without_holidays['date']=carbon_levels_without_holidays.index.date
#threshold = 22
#high_temp_data = carbon_levels_without_holidays[carbon_levels_without_holidays[weather_temp_column] > threshold]
# Iterate over each temperature column

for col in temp_columns:
    #find rows where this sensor exceeds the threshold
    high_temp_data = carbon_levels_without_holidays[carbon_levels_without_holidays[col] > inside_threshold]

    #check if we have any rows where the temperature exceeds 78F
    if not high_temp_data.empty:
        #only want to print one instance per day
        high_temp_data_daily = high_temp_data.groupby('date').head(1)

        #print the date and the sensor column if the condition is met
        print(f"Sensor: {col}")
        print(high_temp_data_daily[['date', col]])  #only print date and the sensor value
        print("-" * 40)  #separator for clarity
        '''
indoor_columns = [col for col in high_temp_data.columns if not ('Outside' in col)]


high_temp_data_indoor = high_temp_data[indoor_columns]

temp_columns = [col for col in temp_columns if col in high_temp_data_indoor.columns]

high_temp_data_clean = high_temp_data_indoor.dropna(subset=temp_columns, how='all')

high_temp_data_clean.loc[:,'hottest_room'] = high_temp_data_clean[temp_columns].idxmax(axis = 1)
high_temp_data_clean.loc[:,'max_temperature'] = high_temp_data_clean[temp_columns].max(axis = 1)

print(high_temp_data_clean[['hottest_room', 'max_temperature', weather_temp_column]])
'''
'''
#print(find_max_temperatures(carbon_levels_without_holidays))
morning = ~np.isin(carbon_levels_without_holidays.index.time, (pd.date_range("8:00", "9:30", freq = "1min").time)) #This will be used to denote the time periods in early morning
carbon_levels_test = carbon_levels_without_holidays[~morning]
#print(carbon_levels_test)
new_name = input("Enter a sensor name to check: ")
if (carbon_levels_test[new_name].quantile(0.65) < 68):
    print("Bad system.")
    print(carbon_levels_test[new_name].quantile(0.65))
else:
    print("Good system with a temp of " + str(carbon_levels_test[new_name].quantile(0.65)) + " degrees Fahrenheit.")
print("AC Need")
print(find_rooms_needing_AC(carbon_levels_without_holidays))
print(carbon_levels_without_holidays)
print(flag_sensors(carbon_levels_test))

high_list, medium_list, light_list = sort_sensors(carbon_levels_without_holidays)

print("High sensors are " + str(high_list))
print(medium_list)

print("Broken sensors: ")
print(broken_sensors(carbon_levels_without_holidays))


while not terminate_loop:
    column_name = input("Enter a sensor name to check if its system is all-set: ")
    filter_level = input("Checking system for T or CO2: ").lower()
    if filter_level == "t":
        if column_name in carbon_levels_without_holidays.columns:
            if ((abs(carbon_levels_without_holidays[column_name].quantile(0.15)-targ1) < 2) and (abs(carbon_levels_without_holidays[column_name].quantile(0.85)- targ2) < 2)):
                print("Good system")
                print("\nThis system runs from " + str(carbon_levels_without_holidays[column_name].quantile(0.15)) + " F to " + str(carbon_levels_without_holidays[column_name].quantile(0.85)) + " F.")
                while v != "yes":
                    v = input("Check again? ").lower()
                    if v == "no":
                        terminate_loop = True
                        break
                    elif v != "yes":
                        print("Invalid, try again: ")

            else:
                print("Bad system")
                print("\nThis system runs from " + str(carbon_levels_without_holidays[column_name].quantile(0.15)) + " F to " + str(carbon_levels_without_holidays[column_name].quantile(0.85)) + " F.")
                while v != "yes":
                    v = input("Check again? ").lower()
                    if v == "no":
                        terminate_loop = True
                        break 
                    elif v != "yes":
                        print("Invalid, try again: ")
    elif filter_level == "co2":
        if column_name in carbon_levels_without_holidays.columns:
            if ((carbon_levels_without_holidays[column_name].quantile(0.45)) < 800):
                print("Good system")
                print("\nThis system runs from " + str(carbon_levels_without_holidays[column_name].quantile(0.45)) + " ppm to " + str(carbon_levels_without_holidays[column_name].quantile(0.85)) + " ppm.")
                while v != "yes":
                    v = input("Check again? ").lower()
                    if v == "no":
                        terminate_loop = True
                        break
                    elif v != "yes":
                        print("Invalid, try again: ")

            else:
                print("Bad system")
                print("\nThis system runs from " + str(carbon_levels_without_holidays[column_name].quantile(0.45)) + " ppm to " + str(carbon_levels_without_holidays[column_name].quantile(0.85)) + " ppm.")
                while v != "yes":
                    v = input("Check again? ").lower()
                    if v == "no":
                        terminate_loop = True
                        break 
                    elif v != "yes":
                        print("Invalid, try again: ")

'''




        





                

