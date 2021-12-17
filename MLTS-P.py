#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:35:05 2021

@author: lee
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import holidays
import datetime
#%%
def get_hour(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return round((int(h) * 3600 + int(m) * 60 + int(s))/3600,2)

def time_until_end_of_day(dt):
    # type: (datetime.datetime) -> datetime.timedelta
    tomorrow = dt + datetime.timedelta(days=1)
    return datetime.datetime.combine(tomorrow, datetime.time.min) - dt

def total_charging_time(dt1,dt2):
    charging_duration = dt2 - dt1
    return charging_duration.seconds/60

def time_until_next_hour(dt):
    next_hour = dt + timedelta(hours=1)-timedelta(minutes=dt.minute)
    left = next_hour - dt
    return left.seconds/60

def time_before_last_hour(dt):
    last_hour = dt - timedelta(minutes=dt.minute) - timedelta(seconds=dt.second) 
    left = dt - last_hour
    if left.seconds == 0:
        return 60
    else:
        return left.seconds/60

def create_label(dt, hour):
    label = str(dt.year) + '/'+str(dt.month) + '/'+str(dt.day) + '-' + str(hour)
    return label

def hours_between(df1,df2):
    if df2.hour==0:
        hours = list(range(df1.hour,24))
    else:
        hours = list(range(df1.hour,df2.hour+1))
    return hours

#%%
df=pd.read_csv("/Users/lee/Documents/Python/MLTS_project/Charging_Behavior_Dataset/Electric_Vehicle_Charging_Station_Energy_Consumption.csv",sep=",")

df.drop(columns=['Address','City','State_Province','Zip_Postal_Code','Start_Time_Zone','End_Time_Zone','Total_Duration__hh_mm_ss_','GHG_Savings__kg_','Gasoline_Savings__gallons_','Port_Type','ObjectId'],inplace=True)
df.rename(columns={"Start_Date___Time":"Plugin_Time","End_Date___Time":"Plugout_Time","Charging_Time__hh_mm_ss_":"Charging_Time","Energy__kWh_":"Energy_kWh"},inplace=True)
df.dropna(subset=['Plugout_Time'],inplace=True)
#%%
#preprocessing the columns 'Plugin_Time', 'Plugout_Time'
#change date value to datetime format
for i in ['Plugin_Time', 'Plugout_Time']:
    df[i] = pd.to_datetime(df[i], infer_datetime_format=True)
#changing the unit to hours for "Charging_Time_h" 
df["Charging_Time_h"] = df.Charging_Time.apply(lambda x: get_hour(x))
#%%
#dealing with overnight data
df_overnight=df[df.Plugin_Time.apply(lambda x:time_until_end_of_day(x).seconds/3600)<df.Charging_Time_h.apply(lambda x:x)]
df_overnight["time_left"] = df_overnight.Plugin_Time.apply(lambda x:time_until_end_of_day(x).seconds/3600)
df_overnight.Plugin_Time=df_overnight.Plugin_Time.apply(lambda x:x+time_until_end_of_day(x))
df_overnight.Energy_kWh=df_overnight.Energy_kWh/df_overnight.Charging_Time_h*(df_overnight.Charging_Time_h-df_overnight.time_left)

for i in df_overnight.index:
    df_overnight["Plugout_Time"][i] = df_overnight.Plugin_Time[i]+timedelta(hours=df_overnight.Charging_Time_h[i]- df_overnight.time_left[i])
df_overnight.rename(columns={"Plugout_Time":"End_Charging_Time"},inplace=True)
#%% 
#modifing value in the original dataframe 
Charging_Time = []
Plugin_Time = list(df.Plugin_Time)
for i in df.Charging_Time_h:
    Charging_Time.append(timedelta(hours=i))
df["End_Charging_Time"] =  np.array(Charging_Time) + np.array(Plugin_Time)         
for i in df_overnight.index:
    df.End_Charging_Time.loc[i] = df.Plugin_Time.loc[i] +time_until_end_of_day(df.Plugin_Time.loc[i])
    df.Energy_kWh.loc[i]=df.Energy_kWh.loc[i]/df.Charging_Time_h.loc[i]*df_overnight.time_left.loc[i]
df_overnight.drop(columns=['Station_Name',  'Charging_Time', 'Charging_Time_h', 'time_left'],inplace=True)
df.drop(columns=['Station_Name', 'Plugout_Time', 'Charging_Time', 'Charging_Time_h'],inplace=True)
#%%
#combining overnight dataframe and original dataframe
df_new = df.append(df_overnight,ignore_index=True)
df_new = df_new.drop(df_new[df_new.Energy_kWh==0].index)
df_new = df_new.reset_index(drop=True)
#%%
#dict with date-hour as key and list of energy demand per each charging behavior
dict_1 = {}
for i in range(30275):
    hours = hours_between(df_new.Plugin_Time[i],df_new.End_Charging_Time[i])
    charging_duration = total_charging_time(df_new.Plugin_Time[i],df_new.End_Charging_Time[i])
    for hour in hours :   
        if hour == min(hours):
            if create_label(df_new.Plugin_Time[i], hour) in dict_1:
                dict_1[create_label(df_new.Plugin_Time[i], hour)].append(round(df_new.Energy_kWh[i]*time_until_next_hour(df_new.Plugin_Time[i])/charging_duration,2))
            else:
                dict_1[create_label(df_new.Plugin_Time[i], hour)] = []
                dict_1[create_label(df_new.Plugin_Time[i], hour)].append(round(df_new.Energy_kWh[i]*time_until_next_hour(df_new.Plugin_Time[i])/charging_duration,2))
        elif hour == max(hours):
            if create_label(df_new.Plugin_Time[i], hour) in dict_1:
                dict_1[create_label(df_new.Plugin_Time[i], hour)].append(round(df_new.Energy_kWh[i]*time_before_last_hour(df_new.End_Charging_Time[i])/charging_duration,2))
            else:
                dict_1[create_label(df_new.Plugin_Time[i], hour)] = []
                dict_1[create_label(df_new.Plugin_Time[i], hour)].append(round(df_new.Energy_kWh[i]*time_before_last_hour(df_new.End_Charging_Time[i])/charging_duration,2))
        else:
            if create_label(df_new.Plugin_Time[i], hour) in dict_1:
                dict_1[create_label(df_new.Plugin_Time[i], hour)].append(round(df_new.Energy_kWh[i]*60/charging_duration,2))
            else:
                dict_1[create_label(df_new.Plugin_Time[i], hour)] = []
                dict_1[create_label(df_new.Plugin_Time[i], hour)].append(round(df_new.Energy_kWh[i]*60/charging_duration,2))
#%%
#dict with date-hour as key and value of energy demand within this hour
dict_2={}
for key,value in dict_1.items():
    dict_2[key] = sum(value)
#%% 
#dataframe for model input
df_model = pd.DataFrame.from_dict(dict_2,orient='index',columns=['Total_Energy_kWh']) 
df_model.reset_index(level=0, inplace=True)
df_model.rename(columns={"index":"Label"},inplace=True)
#Add further detail to the df_model dataframe
#Add data about holiday
# Select country
us_holidays = holidays.US()
# If it is a holidays then it returns True else False
df_model["Holiday"] = df_model.Label.apply(lambda x: datetime.datetime.strptime(x.split('-')[0], "%Y/%m/%d") in us_holidays and datetime.datetime.strptime(x.split('-')[0], "%Y/%m/%d").strftime('%A')=="Saturday" or datetime.datetime.strptime(x.split('-')[0], "%Y/%m/%d").strftime('%A')=="Sunday")











                             
                             