import pandas as pd
import numpy as np
import re

########################################
# Define data cleaning functions

def clean_c_charge_degree(df):
    df['c_charge_degree'] = df['c_charge_degree'].str.replace(r'\(|\)', '', regex=True)
    return df

def extract_year(date):
    return pd.to_datetime(date).year

def extract_month(date):
    return pd.to_datetime(date).month

def process_dates(df):
    df['dob_year'] = df['dob'].apply(extract_year)
    df['c_jail_year'] = df['c_jail_in'].apply(extract_year)
    df['c_jail_month'] = df['c_jail_in'].apply(extract_month)
    return df.drop(columns=['dob', 'c_jail_in'])

def agrupar_tipo_crime(descricao):
    if pd.isna(descricao):
        return 'other'
    descricao = descricao.lower()
    if 'battery' in descricao or 'assault' in descricao or 'violence' in descricao or 'murder' in descricao or 'batt' in descricao:
        return 'violence'
    elif 'theft' in descricao or 'burglary' in descricao or 'robbery' in descricao:
        return 'robbery'
    elif 'drug' in descricao or 'possession' in descricao or 'trafficking' in descricao or 'poss' in descricao or 'cocaine' in descricao or 'heroin' in descricao or 'deliver' in descricao or 'traffick' in descricao:
        return 'drugs'
    elif 'driving' in descricao or 'traffic' in descricao or 'license' in descricao or 'driv' in descricao or 'vehicle' in descricao or 'conduct' in descricao:
        return 'traffic'
    else:
        return 'other'

def group_races(df):
    race_map = df['race'].value_counts()
    common_races = race_map[race_map >= 50].index.tolist()
    df['race_grouped'] = df['race'].apply(lambda x: x if x in common_races else 'Other')
    return df.drop(columns=['race'])

def clean_data(df):
    # Drop columns
    df = df.drop(columns=['id', 'name', 'c_case_number', 'c_offense_date', 'c_arrest_date'])
    
    # Apply custom transformations
    df = clean_c_charge_degree(df)
    df['c_charge_desc'] = df['c_charge_desc'].apply(agrupar_tipo_crime)
    # Group races
    df = group_races(df)
    
    # Convert to categorical
    df['c_charge_desc'] = pd.Categorical(df['c_charge_desc'], categories=['violence', 'robbery', 'drugs', 'traffic', 'other'])
    df['sex'] = df['sex'].astype('category')
    df['race_grouped'] = df['race_grouped'].astype('category')
    df['c_charge_degree'] = df['c_charge_degree'].astype('category')
    
    # Process dates
    df = process_dates(df)
    
    return df

# End data cleaning functions
########################################
