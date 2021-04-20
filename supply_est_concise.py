# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:47:19 2020

This script imports and analyzes Massachusetts' PUMS data 2014-2018, 
ACS tables 2014-2018 at census tract level, and Infogroup 2016 business data
to identify locations of immigrant communities that are also host to substantial
concentrations of immigrant entrepreneurs who own their own business.

Analysis steps inspired by https://stharrold.github.io/20160110-etl-census-with-python.html

PUMS fields:
    PUMA: PUMA code.
    SPORDER: Person number
    REGION: Region code (Northeast == 1)
    ST: State code (MA == 25)
    PWGTP: Person weight (integer)
    WGTP: Housing unit weight (also WGTP1 through WGTP80 for replicate weights)
    ADJHSG: Factor for adjusting $ amounts to 2018 $$
    NP: Number of persons in household (we select 1, 2)
    TYPE: Type of housing unit (we select 1 = housing unit, not group or institutional)
    BDSP: Number of bedrooms (we select 2 and above)
    RMSP: Number of rooms (numeric), up to 99
    BLD: Units in structure (2chr): 
        01 = mobile home/trailer
        02 = single family home, detached
        03 = single-family home, attached
        04 = 2 apartments
        05 = 3-4 apartments
        06 = 5-9 apartments
        07 = 10-19 apartments
        08 = 20-49 apartments
        09 = 50+ apartments
        10 = boat, RB, van, etc.
    RNTP: Monthly rent amount
    GRNTP: Gross rent amount (bbbbb), monthly
    CONP (numeric)): Condo fee, monthly $$
    ELEFP (char): Electricity cost flag: 
            b = N/A, GQ vacant
            1 = included in condo fee
            2 = no charge or no electricity in use
            3 = valid monthly electicity cost
    ELEP: Monthly electricity cost in $ (bbb = no charge for some reason, numeric = dollars)
    FULFP: Fuel cost flag var (see ELEFP)
    FULP: Fuel cost yearly, 4 digits
    GASFP: Gas cost flag
        b: N/A because GQ or vacant
        1: included in rent or condo fee
        2: included in electric payment
        3: no charge or gas not in use
        4: monthly gas charge exists
        
        GASP: Gas cost monthly (bbb or 0-999)
    MRGP: First mortgage payment (Monthly, bbbbb)
    SMP: Total payment on all 2nd junior morgages, home equity loans
    TEN: Tenure (ie owned with mortgage (1), owned free and clear (2), rented, occupied rent-free)
    FES: Family type and employment status. Heterosexual couples and non-families only. 
    HHT: Household/family type
        N/A: GQ/vacant
        1: Married couple
        2: Other family householde: Male householder with no spouse present
        3: Other family household: Female householder with no spouse present
        4: Nonfamily: Male householder living alone
        5: Nonfamily: Male householder not living alone
        6: Nonfamily: Female householder living alone
        7: Female householder not living alone
    HINCP: Household Income (past 12 monthls): bbbbbbbb, includes loss options
    PARTNER: Unmarried partner households
        b: N/A (GQ/vacant)
        1: no unmarried partner in household
        2: Male householder, male partner
        3: Male householder, female partner
        4: Female householder, female partner
        5: Female householder, male partner
    R60: Presence of persons 60 years and over in household (unweighted)
        b: N/A because GQ or vacant
        0: No
        1: Yes, one person
        2: Yes, 2+ people
    R65: Presence of persons 65 years and over in household (unweighted)
        See r60
    WIF: Workers in family during past 12 months
        bb: GQ/vacant/not a family
        01: Householder, spouse worked full time
        02, 03, 04, 05, 06, 07, 08: Some combination of no work, part time work, full time work
        10, 11, 13, 14: Full or part time work, no spouse present
        12, 15: Householder did not work, no spouse prsent
    
        
    


@author: cspence
"""


# import os
# import sys
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns

# PUMS analysis
ma_hhfile = "K:\\DataServices\\Datasets\\U.S. Census and Demographics\\PUMS\\Raw\\pums_2014_18\\csv_hma\\psam_h25.csv"
ma_hhpums = pd.read_csv(ma_hhfile, low_memory = False)

ma_pumafile = "K:\\DataServices\\Projects\\Current_projects\\Housing\\Intergenerational_Homesharing\\Data\\Tabular\\justpumas.csv"
ma_pumas = pd.read_csv(ma_pumafile, low_memory = False)
ma_pumas = ma_pumas[['puma5', 'puma_name']]

ma_hhpums = ma_hhpums.join(ma_pumas.set_index('puma5'), on='PUMA')

# Select only one or two person households
ma_housingu = ma_hhpums.loc[ma_hhpums['TYPE'] == 1]
# Sub-select OWNED housing units only (new as of )
ma_ownedunit = ma_housingu.loc[ma_housingu['TEN'].isin([1,2])]
# One person households
ma_onepersonhh = ma_ownedunit.loc[ma_ownedunit['NP'] == 1]
# Two bedrooms (one extra)
ma_onepersonhh_2br = ma_onepersonhh.loc[~ma_onepersonhh['BDSP'].isin(['bb',1.0])]
ma_onep_2br_o60 = ma_onepersonhh_2br.loc[ma_onepersonhh_2br['R60'] == 1]
ma_onep_2br_o65 = ma_onepersonhh_2br.loc[ma_onepersonhh_2br['R65'] == 1]
# Cost-burdened (30%)
ma_onep_2br_o60_cb30 = ma_onep_2br_o60.loc[ma_onep_2br_o60['OCPIP'] >= 30.0]
ma_onep_2br_o65_cb30 = ma_onep_2br_o65.loc[ma_onep_2br_o65['OCPIP'] >= 30.0]
# Cost-burdened (50%)
ma_onep_2br_o60_cb50 = ma_onep_2br_o60.loc[ma_onep_2br_o60['OCPIP'] >= 50.0]
ma_onep_2br_o65_cb50 = ma_onep_2br_o65.loc[ma_onep_2br_o65['OCPIP'] >= 50.0]
# 3 bedrooms (2 extra)
ma_onepersonhh_3br = ma_onepersonhh.loc[~ma_onepersonhh['BDSP'].isin(['bb',1.0, 2.0])]
ma_onep_3br_o60 = ma_onepersonhh_3br.loc[ma_onepersonhh_3br['R60'] == 1]
ma_onep_3br_o65 = ma_onepersonhh_3br.loc[ma_onepersonhh_3br['R65'] == 1]
# Cost-burdened (30%)
ma_onep_3br_o60_cb30 = ma_onep_3br_o60.loc[ma_onep_3br_o60['OCPIP'] >= 30.0]
ma_onep_3br_o65_cb30 = ma_onep_3br_o65.loc[ma_onep_3br_o65['OCPIP'] >= 30.0]
# Cost-burdened (50%)
ma_onep_3br_o60_cb50 = ma_onep_3br_o60.loc[ma_onep_3br_o60['OCPIP'] >= 50.0]
ma_onep_3br_o65_cb50 = ma_onep_3br_o65.loc[ma_onep_3br_o65['OCPIP'] >= 50.0]

# Two person couple households
ma_twopersonhh = ma_ownedunit.loc[ma_ownedunit['NP'] == 2]
ma_twopersonhh = ma_twopersonhh.loc[(ma_twopersonhh['PARTNER'].isin([2.0, 3.0, 4.0, 5.0]) | ma_twopersonhh['HHT'] == 1.0) | ma_twopersonhh['SSMC'].isin([1.0,2.0])]
# Two bedrooms (one extra)
ma_twopersonhh_2br = ma_twopersonhh.loc[~ma_twopersonhh['BDSP'].isin(['bb',1.0])]
ma_twop_2br_o60 = ma_twopersonhh_2br.loc[ma_twopersonhh_2br['R60'] == 2]
ma_twop_2br_o65 = ma_twopersonhh_2br.loc[ma_twopersonhh_2br['R65'] == 2]
# Cost-burdened (30%)
ma_twop_2br_o60_cb30 = ma_twop_2br_o60.loc[ma_twop_2br_o60['OCPIP'] >= 30.0]
ma_twop_2br_o65_cb30 = ma_twop_2br_o65.loc[ma_twop_2br_o65['OCPIP'] >= 30.0]
# Cost-burdened (50%)
ma_twop_2br_o60_cb50 = ma_twop_2br_o60.loc[ma_twop_2br_o60['OCPIP'] >= 50.0]
ma_twop_2br_o65_cb50 = ma_twop_2br_o65.loc[ma_twop_2br_o65['OCPIP'] >= 50.0]
# 3 bedrooms (2 extra)
ma_twopersonhh_3br = ma_twopersonhh.loc[~ma_twopersonhh['BDSP'].isin(['bb',1.0, 2.0])]
ma_twop_3br_o60 = ma_twopersonhh_3br.loc[ma_twopersonhh_3br['R60'] == 2]
ma_twop_3br_o65 = ma_twopersonhh_3br.loc[ma_twopersonhh_3br['R65'] == 2]
# Cost-burdened (30%)
ma_twop_3br_o60_cb30 = ma_twop_3br_o60.loc[ma_twop_3br_o60['OCPIP'] >= 30.0]
ma_twop_3br_o65_cb30 = ma_twop_3br_o65.loc[ma_twop_3br_o65['OCPIP'] >= 30.0]
# Cost-burdened (50%)
ma_twop_3br_o60_cb50 = ma_twop_3br_o60.loc[ma_twop_3br_o60['OCPIP'] >= 50.0]
ma_twop_3br_o65_cb50 = ma_twop_3br_o65.loc[ma_twop_3br_o65['OCPIP'] >= 50.0]


# single-person AND Two person couple households
ma_1or2phh = ma_twopersonhh.append(ma_onepersonhh, ignore_index=True)
# Two bedrooms (one extra)
ma_1or2phh_2br = ma_1or2phh.loc[~ma_1or2phh['BDSP'].isin(['bb',1.0])]
ma_1or2p_2br_o60 = ma_1or2phh_2br.loc[ma_1or2phh_2br['R60'] == 2]
ma_1or2p_2br_o65 = ma_1or2phh_2br.loc[ma_1or2phh_2br['R65'] == 2]
# Cost-burdened (30%)
ma_1or2p_2br_o60_cb30 = ma_1or2p_2br_o60.loc[ma_1or2p_2br_o60['OCPIP'] >= 30.0]
ma_1or2p_2br_o65_cb30 = ma_1or2p_2br_o65.loc[ma_1or2p_2br_o65['OCPIP'] >= 30.0]
# Cost-burdened (50%)
ma_1or2p_2br_o60_cb50 = ma_1or2p_2br_o60.loc[ma_1or2p_2br_o60['OCPIP'] >= 50.0]
ma_1or2p_2br_o65_cb50 = ma_1or2p_2br_o65.loc[ma_1or2p_2br_o65['OCPIP'] >= 50.0]
# 3 bedrooms (2 extra)
ma_1or2phh_3br = ma_1or2phh.loc[~ma_1or2phh['BDSP'].isin(['bb',1.0, 2.0])]
ma_1or2p_3br_o60 = ma_1or2phh_3br.loc[ma_1or2phh_3br['R60'] == 2]
ma_1or2p_3br_o65 = ma_1or2phh_3br.loc[ma_1or2phh_3br['R65'] == 2]
# Cost-burdened (30%)
ma_1or2p_3br_o60_cb30 = ma_1or2p_3br_o60.loc[ma_1or2p_3br_o60['OCPIP'] >= 30.0]
ma_1or2p_3br_o65_cb30 = ma_1or2p_3br_o65.loc[ma_1or2p_3br_o65['OCPIP'] >= 30.0]
# Cost-burdened (50%)
ma_1or2p_3br_o60_cb50 = ma_1or2p_3br_o60.loc[ma_1or2p_3br_o60['OCPIP'] >= 50.0]
ma_1or2p_3br_o65_cb50 = ma_1or2p_3br_o65.loc[ma_1or2p_3br_o65['OCPIP'] >= 50.0]


# Estimate number by PUMA
PUMAs_ma = ma_hhpums['PUMA'].unique()
PUMAs_study = [3301, 3303, 3302, 3305, 3304, 506, 507]

hhs_all = np.empty(len(PUMAs_study))
hh1p60o_2r = np.empty(len(PUMAs_study))
hh1p60o_3r = np.empty(len(PUMAs_study))
hh2p60o_2r = np.empty(len(PUMAs_study))
hh2p60o_3r = np.empty(len(PUMAs_study))
hh12p60o_2r = np.empty(len(PUMAs_study))
hh12p60o_3r = np.empty(len(PUMAs_study))
hh1p65o_2r = np.empty(len(PUMAs_study))
hh1p65o_3r = np.empty(len(PUMAs_study))
hh2p65o_2r = np.empty(len(PUMAs_study))
hh2p65o_3r = np.empty(len(PUMAs_study))
hh12p65o_2r = np.empty(len(PUMAs_study))
hh12p65o_3r = np.empty(len(PUMAs_study))
hh1p60o_2r_cb30 = np.empty(len(PUMAs_study))
hh1p60o_3r_cb30 = np.empty(len(PUMAs_study))
hh2p60o_2r_cb30 = np.empty(len(PUMAs_study))
hh2p60o_3r_cb30 = np.empty(len(PUMAs_study))
hh12p60o_2r_cb30 = np.empty(len(PUMAs_study))
hh12p60o_3r_cb30 = np.empty(len(PUMAs_study))
hh1p65o_2r_cb30 = np.empty(len(PUMAs_study))
hh1p65o_3r_cb30 = np.empty(len(PUMAs_study))
hh2p65o_2r_cb30 = np.empty(len(PUMAs_study))
hh2p65o_3r_cb30 = np.empty(len(PUMAs_study))
hh12p65o_2r_cb30 = np.empty(len(PUMAs_study))
hh12p65o_3r_cb30 = np.empty(len(PUMAs_study))
hh1p60o_2r_cb50 = np.empty(len(PUMAs_study))
hh1p60o_3r_cb50 = np.empty(len(PUMAs_study))
hh2p60o_2r_cb50 = np.empty(len(PUMAs_study))
hh2p60o_3r_cb50 = np.empty(len(PUMAs_study))
hh12p60o_2r_cb50 = np.empty(len(PUMAs_study))
hh12p60o_3r_cb50 = np.empty(len(PUMAs_study))
hh1p65o_2r_cb50 = np.empty(len(PUMAs_study))
hh1p65o_3r_cb50 = np.empty(len(PUMAs_study))
hh2p65o_2r_cb50 = np.empty(len(PUMAs_study))
hh2p65o_3r_cb50 = np.empty(len(PUMAs_study))
hh12p65o_2r_cb50 = np.empty(len(PUMAs_study))
hh12p65o_3r_cb50 = np.empty(len(PUMAs_study))

hhs_allmoe = np.empty(len(PUMAs_study))
hh1p60o_2rmoe = np.empty(len(PUMAs_study))
hh1p60o_3rmoe = np.empty(len(PUMAs_study))
hh2p60o_2rmoe = np.empty(len(PUMAs_study))
hh2p60o_3rmoe = np.empty(len(PUMAs_study))
hh12p60o_2rmoe = np.empty(len(PUMAs_study))
hh12p60o_3rmoe = np.empty(len(PUMAs_study))
hh1p65o_2rmoe = np.empty(len(PUMAs_study))
hh1p65o_3rmoe = np.empty(len(PUMAs_study))
hh2p65o_2rmoe = np.empty(len(PUMAs_study))
hh2p65o_3rmoe = np.empty(len(PUMAs_study))
hh12p65o_2rmoe = np.empty(len(PUMAs_study))
hh12p65o_3rmoe = np.empty(len(PUMAs_study))
hh1p60o_2r_cb30moe = np.empty(len(PUMAs_study))
hh1p60o_3r_cb30moe = np.empty(len(PUMAs_study))
hh2p60o_2r_cb30moe = np.empty(len(PUMAs_study))
hh2p60o_3r_cb30moe = np.empty(len(PUMAs_study))
hh12p60o_2r_cb30moe = np.empty(len(PUMAs_study))
hh12p60o_3r_cb30moe = np.empty(len(PUMAs_study))
hh1p65o_2r_cb30moe = np.empty(len(PUMAs_study))
hh1p65o_3r_cb30moe = np.empty(len(PUMAs_study))
hh2p65o_2r_cb30moe = np.empty(len(PUMAs_study))
hh2p65o_3r_cb30moe = np.empty(len(PUMAs_study))
hh12p65o_2r_cb30moe = np.empty(len(PUMAs_study))
hh12p65o_3r_cb30moe = np.empty(len(PUMAs_study))
hh1p60o_2r_cb50moe = np.empty(len(PUMAs_study))
hh1p60o_3r_cb50moe = np.empty(len(PUMAs_study))
hh2p60o_2r_cb50moe = np.empty(len(PUMAs_study))
hh2p60o_3r_cb50moe = np.empty(len(PUMAs_study))
hh12p60o_2r_cb50moe = np.empty(len(PUMAs_study))
hh12p60o_3r_cb50moe = np.empty(len(PUMAs_study))
hh1p65o_2r_cb50moe = np.empty(len(PUMAs_study))
hh1p65o_3r_cb50moe = np.empty(len(PUMAs_study))
hh2p65o_2r_cb50moe = np.empty(len(PUMAs_study))
hh2p65o_3r_cb50moe = np.empty(len(PUMAs_study))
hh12p65o_2r_cb50moe = np.empty(len(PUMAs_study))
hh12p65o_3r_cb50moe = np.empty(len(PUMAs_study))

hhs_allmoep = np.empty(len(PUMAs_study))
hh1p60o_2rmoep = np.empty(len(PUMAs_study))
hh1p60o_3rmoep = np.empty(len(PUMAs_study))
hh2p60o_2rmoep = np.empty(len(PUMAs_study))
hh2p60o_3rmoep = np.empty(len(PUMAs_study))
hh12p60o_2rmoep = np.empty(len(PUMAs_study))
hh12p60o_3rmoep = np.empty(len(PUMAs_study))
hh1p65o_2rmoep = np.empty(len(PUMAs_study))
hh1p65o_3rmoep = np.empty(len(PUMAs_study))
hh2p65o_2rmoep = np.empty(len(PUMAs_study))
hh2p65o_3rmoep = np.empty(len(PUMAs_study))
hh12p65o_2rmoep = np.empty(len(PUMAs_study))
hh12p65o_3rmoep = np.empty(len(PUMAs_study))
hh1p60o_2r_cb30moep = np.empty(len(PUMAs_study))
hh1p60o_3r_cb30moep = np.empty(len(PUMAs_study))
hh2p60o_2r_cb30moep = np.empty(len(PUMAs_study))
hh2p60o_3r_cb30moep = np.empty(len(PUMAs_study))
hh12p60o_2r_cb30moep = np.empty(len(PUMAs_study))
hh12p60o_3r_cb30moep = np.empty(len(PUMAs_study))
hh1p65o_2r_cb30moep = np.empty(len(PUMAs_study))
hh1p65o_3r_cb30moep = np.empty(len(PUMAs_study))
hh2p65o_2r_cb30moep = np.empty(len(PUMAs_study))
hh2p65o_3r_cb30moep = np.empty(len(PUMAs_study))
hh12p65o_2r_cb30moep = np.empty(len(PUMAs_study))
hh12p65o_3r_cb30moep = np.empty(len(PUMAs_study))
hh1p60o_2r_cb50moep = np.empty(len(PUMAs_study))
hh1p60o_3r_cb50moep = np.empty(len(PUMAs_study))
hh2p60o_2r_cb50moep = np.empty(len(PUMAs_study))
hh2p60o_3r_cb50moep = np.empty(len(PUMAs_study))
hh12p60o_2r_cb50moep = np.empty(len(PUMAs_study))
hh12p60o_3r_cb50moep = np.empty(len(PUMAs_study))
hh1p65o_2r_cb50moep = np.empty(len(PUMAs_study))
hh1p65o_3r_cb50moep = np.empty(len(PUMAs_study))
hh2p65o_2r_cb50moep = np.empty(len(PUMAs_study))
hh2p65o_3r_cb50moep = np.empty(len(PUMAs_study))
hh12p65o_2r_cb50moep = np.empty(len(PUMAs_study))
hh12p65o_3r_cb50moep = np.empty(len(PUMAs_study))

hhs_allu = np.empty(len(PUMAs_study))
hh1p60o_2ru = np.empty(len(PUMAs_study))
hh1p60o_3ru = np.empty(len(PUMAs_study))
hh2p60o_2ru = np.empty(len(PUMAs_study))
hh2p60o_3ru = np.empty(len(PUMAs_study))
hh12p60o_2ru = np.empty(len(PUMAs_study))
hh12p60o_3ru = np.empty(len(PUMAs_study))
hh1p65o_2ru = np.empty(len(PUMAs_study))
hh1p65o_3ru = np.empty(len(PUMAs_study))
hh2p65o_2ru = np.empty(len(PUMAs_study))
hh2p65o_3ru = np.empty(len(PUMAs_study))
hh12p65o_2ru = np.empty(len(PUMAs_study))
hh12p65o_3ru = np.empty(len(PUMAs_study))
hh1p60o_2r_cb30u = np.empty(len(PUMAs_study))
hh1p60o_3r_cb30u = np.empty(len(PUMAs_study))
hh2p60o_2r_cb30u = np.empty(len(PUMAs_study))
hh2p60o_3r_cb30u = np.empty(len(PUMAs_study))
hh12p60o_2r_cb30u = np.empty(len(PUMAs_study))
hh12p60o_3r_cb30u = np.empty(len(PUMAs_study))
hh1p65o_2r_cb30u = np.empty(len(PUMAs_study))
hh1p65o_3r_cb30u = np.empty(len(PUMAs_study))
hh2p65o_2r_cb30u = np.empty(len(PUMAs_study))
hh2p65o_3r_cb30u = np.empty(len(PUMAs_study))
hh12p65o_2r_cb30u = np.empty(len(PUMAs_study))
hh12p65o_3r_cb30u = np.empty(len(PUMAs_study))
hh1p60o_2r_cb50u = np.empty(len(PUMAs_study))
hh1p60o_3r_cb50u = np.empty(len(PUMAs_study))
hh2p60o_2r_cb50u = np.empty(len(PUMAs_study))
hh2p60o_3r_cb50u = np.empty(len(PUMAs_study))
hh12p60o_2r_cb50u = np.empty(len(PUMAs_study))
hh12p60o_3r_cb50u = np.empty(len(PUMAs_study))
hh1p65o_2r_cb50u = np.empty(len(PUMAs_study))
hh1p65o_3r_cb50u = np.empty(len(PUMAs_study))
hh2p65o_2r_cb50u = np.empty(len(PUMAs_study))
hh2p65o_3r_cb50u = np.empty(len(PUMAs_study))
hh12p65o_2r_cb50u = np.empty(len(PUMAs_study))
hh12p65o_3r_cb50u = np.empty(len(PUMAs_study))

hhs_alll = np.empty(len(PUMAs_study))
hh1p60o_2rl = np.empty(len(PUMAs_study))
hh1p60o_3rl = np.empty(len(PUMAs_study))
hh2p60o_2rl = np.empty(len(PUMAs_study))
hh2p60o_3rl = np.empty(len(PUMAs_study))
hh12p60o_2rl = np.empty(len(PUMAs_study))
hh12p60o_3rl = np.empty(len(PUMAs_study))
hh1p65o_2rl = np.empty(len(PUMAs_study))
hh1p65o_3rl = np.empty(len(PUMAs_study))
hh2p65o_2rl = np.empty(len(PUMAs_study))
hh2p65o_3rl = np.empty(len(PUMAs_study))
hh12p65o_2rl = np.empty(len(PUMAs_study))
hh12p65o_3rl = np.empty(len(PUMAs_study))
hh1p60o_2r_cb30l = np.empty(len(PUMAs_study))
hh1p60o_3r_cb30l = np.empty(len(PUMAs_study))
hh2p60o_2r_cb30l = np.empty(len(PUMAs_study))
hh2p60o_3r_cb30l = np.empty(len(PUMAs_study))
hh12p60o_2r_cb30l = np.empty(len(PUMAs_study))
hh12p60o_3r_cb30l = np.empty(len(PUMAs_study))
hh1p65o_2r_cb30l = np.empty(len(PUMAs_study))
hh1p65o_3r_cb30l = np.empty(len(PUMAs_study))
hh2p65o_2r_cb30l = np.empty(len(PUMAs_study))
hh2p65o_3r_cb30l = np.empty(len(PUMAs_study))
hh12p65o_2r_cb30l = np.empty(len(PUMAs_study))
hh12p65o_3r_cb30l = np.empty(len(PUMAs_study))
hh1p60o_2r_cb50l = np.empty(len(PUMAs_study))
hh1p60o_3r_cb50l = np.empty(len(PUMAs_study))
hh2p60o_2r_cb50l = np.empty(len(PUMAs_study))
hh2p60o_3r_cb50l = np.empty(len(PUMAs_study))
hh12p60o_2r_cb50l = np.empty(len(PUMAs_study))
hh12p60o_3r_cb50l = np.empty(len(PUMAs_study))
hh1p65o_2r_cb50l = np.empty(len(PUMAs_study))
hh1p65o_3r_cb50l = np.empty(len(PUMAs_study))
hh2p65o_2r_cb50l = np.empty(len(PUMAs_study))
hh2p65o_3r_cb50l = np.empty(len(PUMAs_study))
hh12p65o_2r_cb50l = np.empty(len(PUMAs_study))
hh12p65o_3r_cb50l = np.empty(len(PUMAs_study))

def pums_est(datapums):
    # datapums = data.loc[data['PUMA'] == PUMAs_ma[k]]
    numpums = np.sum(datapums['WGTP'].values)
    rdatapums = list()
    for j in range(80):
        fieldname = 'WGTP' + str(j + 1)
        
        numpumstemp = (np.sum(datapums[fieldname].values))
        rdatapums.append(np.square(numpumstemp - numpums))
        
    numpumsmoe = 1.645*(np.sqrt((4/80)*np.sum(rdatapums)))
    numpumsmoep = (numpumsmoe/float(numpums))*100.0
    numpumsu = numpums + numpumsmoe
    numpumsl = numpums - numpumsmoe

    return(numpums, numpumsmoe, numpumsmoep, numpumsu, numpumsl)

pumanum = list()
moe = list()
pumanames = list()



for k in range(len(PUMAs_study)):

    hhs_allpums = ma_hhpums.loc[ma_hhpums['PUMA'] == PUMAs_study[k]]
    hh1p60o_2rpums = ma_onep_2br_o60.loc[ma_onep_2br_o60['PUMA'] == PUMAs_study[k]]
    hh1p60o_3rpums = ma_onep_3br_o60.loc[ma_onep_3br_o60['PUMA'] == PUMAs_study[k]]
    hh2p60o_2rpums = ma_twop_2br_o60.loc[ma_twop_2br_o60['PUMA'] == PUMAs_study[k]]
    hh2p60o_3rpums = ma_twop_3br_o60.loc[ma_twop_3br_o60['PUMA'] == PUMAs_study[k]]
    hh12p60o_2rpums = ma_twop_2br_o60.loc[ma_twop_2br_o60['PUMA'] == PUMAs_study[k]]
    hh12p60o_3rpums = ma_twop_3br_o60.loc[ma_twop_3br_o60['PUMA'] == PUMAs_study[k]]
    hh1p65o_2rpums = ma_onep_2br_o65.loc[ma_onep_2br_o65['PUMA'] == PUMAs_study[k]]
    hh1p65o_3rpums = ma_onep_3br_o65.loc[ma_onep_3br_o65['PUMA'] == PUMAs_study[k]]
    hh2p65o_2rpums = ma_twop_2br_o65.loc[ma_twop_2br_o65['PUMA'] == PUMAs_study[k]]
    hh2p65o_3rpums = ma_twop_3br_o65.loc[ma_twop_3br_o65['PUMA'] == PUMAs_study[k]]
    hh12p65o_2rpums = ma_twop_2br_o65.loc[ma_twop_2br_o65['PUMA'] == PUMAs_study[k]]
    hh12p65o_3rpums = ma_twop_3br_o65.loc[ma_twop_3br_o65['PUMA'] == PUMAs_study[k]]
    pumanames.append(hh1p60o_2rpums['puma_name'].values[0])
    
    hh1p60o_2rpums_cb30 = ma_onep_2br_o60_cb30.loc[ma_onep_2br_o60_cb30['PUMA'] == PUMAs_study[k]]
    hh1p60o_3rpums_cb30 = ma_onep_3br_o60_cb30.loc[ma_onep_3br_o60_cb30['PUMA'] == PUMAs_study[k]]
    hh2p60o_2rpums_cb30 = ma_twop_2br_o60_cb30.loc[ma_twop_2br_o60_cb30['PUMA'] == PUMAs_study[k]]
    hh2p60o_3rpums_cb30 = ma_twop_3br_o60_cb30.loc[ma_twop_3br_o60_cb30['PUMA'] == PUMAs_study[k]]
    hh12p60o_2rpums_cb30 = ma_twop_2br_o60_cb30.loc[ma_twop_2br_o60_cb30['PUMA'] == PUMAs_study[k]]
    hh12p60o_3rpums_cb30 = ma_twop_3br_o60_cb30.loc[ma_twop_3br_o60_cb30['PUMA'] == PUMAs_study[k]]
    hh1p65o_2rpums_cb30 = ma_onep_2br_o65_cb30.loc[ma_onep_2br_o65_cb30['PUMA'] == PUMAs_study[k]]
    hh1p65o_3rpums_cb30 = ma_onep_3br_o65_cb30.loc[ma_onep_3br_o65_cb30['PUMA'] == PUMAs_study[k]]
    hh2p65o_2rpums_cb30 = ma_twop_2br_o65_cb30.loc[ma_twop_2br_o65_cb30['PUMA'] == PUMAs_study[k]]
    hh2p65o_3rpums_cb30 = ma_twop_3br_o65_cb30.loc[ma_twop_3br_o65_cb30['PUMA'] == PUMAs_study[k]]
    hh12p65o_2rpums_cb30 = ma_twop_2br_o65_cb30.loc[ma_twop_2br_o65_cb30['PUMA'] == PUMAs_study[k]]
    hh12p65o_3rpums_cb30 = ma_twop_3br_o65_cb30.loc[ma_twop_3br_o65_cb30['PUMA'] == PUMAs_study[k]]
    
    hh1p60o_2rpums_cb50 = ma_onep_2br_o60_cb50.loc[ma_onep_2br_o60_cb50['PUMA'] == PUMAs_study[k]]
    hh1p60o_3rpums_cb50 = ma_onep_3br_o60_cb50.loc[ma_onep_3br_o60_cb50['PUMA'] == PUMAs_study[k]]
    hh2p60o_2rpums_cb50 = ma_twop_2br_o60_cb50.loc[ma_twop_2br_o60_cb50['PUMA'] == PUMAs_study[k]]
    hh2p60o_3rpums_cb50 = ma_twop_3br_o60_cb50.loc[ma_twop_3br_o60_cb50['PUMA'] == PUMAs_study[k]]
    hh12p60o_2rpums_cb50 = ma_twop_2br_o60_cb50.loc[ma_twop_2br_o60_cb50['PUMA'] == PUMAs_study[k]]
    hh12p60o_3rpums_cb50 = ma_twop_3br_o60_cb50.loc[ma_twop_3br_o60_cb50['PUMA'] == PUMAs_study[k]]
    hh1p65o_2rpums_cb50 = ma_onep_2br_o65_cb50.loc[ma_onep_2br_o65_cb50['PUMA'] == PUMAs_study[k]]
    hh1p65o_3rpums_cb50 = ma_onep_3br_o65_cb50.loc[ma_onep_3br_o65_cb50['PUMA'] == PUMAs_study[k]]
    hh2p65o_2rpums_cb50 = ma_twop_2br_o65_cb50.loc[ma_twop_2br_o65_cb50['PUMA'] == PUMAs_study[k]]
    hh2p65o_3rpums_cb50 = ma_twop_3br_o65_cb50.loc[ma_twop_3br_o65_cb50['PUMA'] == PUMAs_study[k]]
    hh12p65o_2rpums_cb50 = ma_twop_2br_o65_cb50.loc[ma_twop_2br_o65_cb50['PUMA'] == PUMAs_study[k]]
    hh12p65o_3rpums_cb50 = ma_twop_3br_o65_cb50.loc[ma_twop_3br_o65_cb50['PUMA'] == PUMAs_study[k]]
    
    (hhs_all[k], hhs_allmoe[k], hhs_allmoep[k], hhs_allu[k], hhs_alll[k]) = pums_est(hhs_allpums)
    (hh1p60o_2r[k], hh1p60o_2rmoe[k], hh1p60o_2rmoep[k], hh1p60o_2ru[k], hh1p60o_2rl[k]) = pums_est(hh1p60o_2rpums)
    (hh1p60o_3r[k], hh1p60o_3rmoe[k], hh1p60o_3rmoep[k], hh1p60o_3ru[k], hh1p60o_3rl[k]) = pums_est(hh1p60o_3rpums)
    (hh2p60o_2r[k], hh2p60o_2rmoe[k], hh2p60o_2rmoep[k], hh2p60o_2ru[k], hh2p60o_2rl[k]) = pums_est(hh2p60o_2rpums)
    (hh2p60o_3r[k], hh2p60o_3rmoe[k], hh2p60o_3rmoep[k], hh2p60o_3ru[k], hh2p60o_3rl[k]) = pums_est(hh2p60o_3rpums)
    (hh12p60o_2r[k], hh12p60o_2rmoe[k], hh12p60o_2rmoep[k], hh12p60o_2ru[k], hh12p60o_2rl[k]) = pums_est(hh12p60o_2rpums)
    (hh12p60o_3r[k], hh12p60o_3rmoe[k], hh12p60o_3rmoep[k], hh12p60o_3ru[k], hh12p60o_3rl[k]) = pums_est(hh12p60o_3rpums)
    (hh1p65o_2r[k], hh1p65o_2rmoe[k], hh1p65o_2rmoep[k], hh1p65o_2ru[k], hh1p65o_2rl[k]) = pums_est(hh1p65o_2rpums)
    (hh1p65o_3r[k], hh1p65o_3rmoe[k], hh1p65o_3rmoep[k], hh1p65o_3ru[k], hh1p65o_3rl[k]) = pums_est(hh1p65o_3rpums)
    (hh2p65o_2r[k], hh2p65o_2rmoe[k], hh2p65o_2rmoep[k], hh2p65o_2ru[k], hh2p65o_2rl[k]) = pums_est(hh2p65o_2rpums)
    (hh2p65o_3r[k], hh2p65o_3rmoe[k], hh2p65o_3rmoep[k], hh2p65o_3ru[k], hh2p65o_3rl[k]) = pums_est(hh2p65o_3rpums)
    (hh12p65o_2r[k], hh12p65o_2rmoe[k], hh12p65o_2rmoep[k], hh12p65o_2ru[k], hh12p65o_2rl[k]) = pums_est(hh12p65o_2rpums)
    (hh12p65o_3r[k], hh12p65o_3rmoe[k], hh12p65o_3rmoep[k], hh12p65o_3ru[k], hh12p65o_3rl[k]) = pums_est(hh12p65o_3rpums)
    
    (hh1p60o_2r_cb30[k], hh1p60o_2r_cb30moe[k], hh1p60o_2r_cb30moep[k], hh1p60o_2r_cb30u[k], hh1p60o_2r_cb30l[k]) = pums_est(hh1p60o_2rpums_cb30)
    (hh1p60o_3r_cb30[k], hh1p60o_3r_cb30moe[k], hh1p60o_3r_cb30moep[k], hh1p60o_3r_cb30u[k], hh1p60o_3r_cb30l[k]) = pums_est(hh1p60o_3rpums_cb30)
    (hh2p60o_2r_cb30[k], hh2p60o_2r_cb30moe[k], hh2p60o_2r_cb30moep[k], hh2p60o_2r_cb30u[k], hh2p60o_2r_cb30l[k]) = pums_est(hh2p60o_2rpums_cb30)
    (hh2p60o_3r_cb30[k], hh2p60o_3r_cb30moe[k], hh2p60o_3r_cb30moep[k], hh2p60o_3r_cb30u[k], hh2p60o_3r_cb30l[k]) = pums_est(hh2p60o_3rpums_cb30)
    (hh12p60o_2r_cb30[k], hh12p60o_2r_cb30moe[k], hh12p60o_2r_cb30moep[k], hh12p60o_2r_cb30u[k], hh12p60o_2r_cb30l[k]) = pums_est(hh12p60o_2rpums_cb30)
    (hh12p60o_3r_cb30[k], hh12p60o_3r_cb30moe[k], hh12p60o_3r_cb30moep[k], hh12p60o_3r_cb30u[k], hh12p60o_3r_cb30l[k]) = pums_est(hh12p60o_3rpums_cb30)
    (hh1p65o_2r_cb30[k], hh1p65o_2r_cb30moe[k], hh1p65o_2r_cb30moep[k], hh1p65o_2r_cb30u[k], hh1p65o_2r_cb30l[k]) = pums_est(hh1p65o_2rpums_cb30)
    (hh1p65o_3r_cb30[k], hh1p65o_3r_cb30moe[k], hh1p65o_3r_cb30moep[k], hh1p65o_3r_cb30u[k], hh1p65o_3r_cb30l[k]) = pums_est(hh1p65o_3rpums_cb30)
    (hh2p65o_2r_cb30[k], hh2p65o_2r_cb30moe[k], hh2p65o_2r_cb30moep[k], hh2p65o_2r_cb30u[k], hh2p65o_2r_cb30l[k]) = pums_est(hh2p65o_2rpums_cb30)
    (hh2p65o_3r_cb30[k], hh2p65o_3r_cb30moe[k], hh2p65o_3r_cb30moep[k], hh2p65o_3r_cb30u[k], hh2p65o_3r_cb30l[k]) = pums_est(hh2p65o_3rpums_cb30)
    (hh12p65o_2r_cb30[k], hh12p65o_2r_cb30moe[k], hh12p65o_2r_cb30moep[k], hh12p65o_2r_cb30u[k], hh12p65o_2r_cb30l[k]) = pums_est(hh12p65o_2rpums_cb30)
    (hh12p65o_3r_cb30[k], hh12p65o_3r_cb30moe[k], hh12p65o_3r_cb30moep[k], hh12p65o_3r_cb30u[k], hh12p65o_3r_cb30l[k]) = pums_est(hh12p65o_3rpums_cb30)
    
    (hh1p60o_2r_cb50[k], hh1p60o_2r_cb50moe[k], hh1p60o_2r_cb50moep[k], hh1p60o_2r_cb50u[k], hh1p60o_2r_cb50l[k]) = pums_est(hh1p60o_2rpums_cb50)
    (hh1p60o_3r_cb50[k], hh1p60o_3r_cb50moe[k], hh1p60o_3r_cb50moep[k], hh1p60o_3r_cb50u[k], hh1p60o_3r_cb50l[k]) = pums_est(hh1p60o_3rpums_cb50)
    (hh2p60o_2r_cb50[k], hh2p60o_2r_cb50moe[k], hh2p60o_2r_cb50moep[k], hh2p60o_2r_cb50u[k], hh2p60o_2r_cb50l[k]) = pums_est(hh2p60o_2rpums_cb50)
    (hh2p60o_3r_cb50[k], hh2p60o_3r_cb50moe[k], hh2p60o_3r_cb50moep[k], hh2p60o_3r_cb50u[k], hh2p60o_3r_cb50l[k]) = pums_est(hh2p60o_3rpums_cb50)
    (hh12p60o_2r_cb50[k], hh12p60o_2r_cb50moe[k], hh12p60o_2r_cb50moep[k], hh12p60o_2r_cb50u[k], hh12p60o_2r_cb50l[k]) = pums_est(hh12p60o_2rpums_cb50)
    (hh12p60o_3r_cb50[k], hh12p60o_3r_cb50moe[k], hh12p60o_3r_cb50moep[k], hh12p60o_3r_cb50u[k], hh12p60o_3r_cb50l[k]) = pums_est(hh12p60o_3rpums_cb50)
    (hh1p65o_2r_cb50[k], hh1p65o_2r_cb50moe[k], hh1p65o_2r_cb50moep[k], hh1p65o_2r_cb50u[k], hh1p65o_2r_cb50l[k]) = pums_est(hh1p65o_2rpums_cb50)
    (hh1p65o_3r_cb50[k], hh1p65o_3r_cb50moe[k], hh1p65o_3r_cb50moep[k], hh1p65o_3r_cb50u[k], hh1p65o_3r_cb50l[k]) = pums_est(hh1p65o_3rpums_cb50)
    (hh2p65o_2r_cb50[k], hh2p65o_2r_cb50moe[k], hh2p65o_2r_cb50moep[k], hh2p65o_2r_cb50u[k], hh2p65o_2r_cb50l[k]) = pums_est(hh2p65o_2rpums_cb50)
    (hh2p65o_3r_cb50[k], hh2p65o_3r_cb50moe[k], hh2p65o_3r_cb50moep[k], hh2p65o_3r_cb50u[k], hh2p65o_3r_cb50l[k]) = pums_est(hh2p65o_3rpums_cb50)
    (hh12p65o_2r_cb50[k], hh12p65o_2r_cb50moe[k], hh12p65o_2r_cb50moep[k], hh12p65o_2r_cb50u[k], hh12p65o_2r_cb50l[k]) = pums_est(hh12p65o_2rpums_cb50)
    (hh12p65o_3r_cb50[k], hh12p65o_3r_cb50moe[k], hh12p65o_3r_cb50moep[k], hh12p65o_3r_cb50u[k], hh12p65o_3r_cb50l[k]) = pums_est(hh12p65o_3rpums_cb50)
    
    
    
    
supplypuma_single60plus = {'PUMA': PUMAs_study,
                           'PUMA Name': pumanames,
                           'All occupied housing units': hhs_all,
                           'All occupied housing units MoE': hhs_allmoe,
                           'All occupied housing units MoE (%)': hhs_allmoep,
                           'All occupied housing units (Lower)': hhs_alll,
                           'All occupied housing units (Upper)': hhs_allu,
                           'At least one extra bedroom': hh1p60o_2r,
                           'At least one extra bedroom MoE': hh1p60o_2rmoe,
                           'At least one extra bedroom MoE (%)': hh1p60o_2rmoep,
                           'At least one extra bedroom (Lower)': hh1p60o_2rl,
                           'At least one extra bedroom (Upper)': hh1p60o_2ru,
                           'At least two extra bedrooms': hh1p60o_3r,
                           'At least two extra bedrooms MoE': hh1p60o_3rmoe,
                           'At least two extra bedrooms MoE (%)': hh1p60o_3rmoep,
                           'At least two extra bedrooms (Lower)': hh1p60o_3rl,
                           'At least two extra bedrooms (Upper)': hh1p60o_3ru,
                           'Cost-burdened (30%) with at least one extra bedroom': hh1p60o_2r_cb30,
                           'Cost-burdened (30%) with at least one extra bedroom MoE': hh1p60o_2r_cb30moe,
                           'Cost-burdened (30%) with at least one extra bedroom MoE (%)': hh1p60o_2r_cb30moep,
                           'Cost-burdened (30%) with at least one extra bedroom (Lower)': hh1p60o_2r_cb30l,
                           'Cost-burdened (30%) with at least one extra bedroom (Upper)': hh1p60o_2r_cb30u,
                           'Cost-burdened (30%) with at least two extra bedrooms': hh1p60o_3r_cb30,
                           'Cost-burdened (30%) with at least two extra bedrooms MoE': hh1p60o_3r_cb30moe,
                           'Cost-burdened (30%) with at least two extra bedrooms MoE (%)': hh1p60o_3r_cb30moep,
                           'Cost-burdened (30%) with at least two extra bedrooms (Lower)': hh1p60o_3r_cb30l,
                           'Cost-burdened (30%) with at least two extra bedrooms (Upper)': hh1p60o_3r_cb30u,
                           'Cost-burdened (50%) with at least one extra bedroom': hh1p60o_2r_cb50,
                           'Cost-burdened (50%) with at least one extra bedroom MoE': hh1p60o_2r_cb50moe,
                           'Cost-burdened (50%) with at least one extra bedroom MoE (%)': hh1p60o_2r_cb50moep,
                           'Cost-burdened (50%) with at least one extra bedroom (Lower)': hh1p60o_2r_cb50l,
                           'Cost-burdened (50%) with at least one extra bedroom (Upper)': hh1p60o_2r_cb50u,
                           'Cost-burdened (50%) with at least two extra bedrooms': hh1p60o_3r_cb50,
                           'Cost-burdened (50%) with at least two extra bedrooms MoE': hh1p60o_3r_cb50moe,
                           'Cost-burdened (50%) with at least two extra bedrooms MoE (%)': hh1p60o_3r_cb50moep,
                           'Cost-burdened (50%) with at least two extra bedrooms (Lower)': hh1p60o_3r_cb50l,
                           'Cost-burdened (50%) with at least two extra bedrooms (Upper)': hh1p60o_3r_cb50u}
                              
supplypuma_couple60plus = {'PUMA': PUMAs_study,
                           'PUMA Name': pumanames,
                           'All occupied housing units': hhs_all,
                           'All occupied housing units MoE': hhs_allmoe,
                           'All occupied housing units MoE (%)': hhs_allmoep,
                           'All occupied housing units (Lower)': hhs_alll,
                           'All occupied housing units (Upper)': hhs_allu,                            
                           'At least one extra bedroom': hh2p60o_2r,
                           'At least one extra bedroom MoE': hh2p60o_2rmoe,
                           'At least one extra bedroom MoE (%)': hh2p60o_2rmoep,
                           'At least one extra bedroom (Lower)': hh2p60o_2rl,
                           'At least one extra bedroom (Upper)': hh2p60o_2ru,
                           'At least two extra bedrooms': hh2p60o_3r,
                           'At least one extra bedroom MoE': hh2p60o_2rmoe,
                           'At least two extra bedrooms MoE (%)': hh2p60o_3rmoep,
                           'At least two extra bedrooms (Lower)': hh2p60o_3rl,
                           'At least two extra bedrooms (Upper)': hh2p60o_3ru,
                           'Cost-burdened (30%) with at least one extra bedroom': hh2p60o_2r_cb30,
                           'Cost-burdened (30%) with at least one extra bedroom MoE': hh2p60o_2r_cb30moe,
                           'Cost-burdened (30%) with at least one extra bedroom MoE (%)': hh2p60o_2r_cb30moep,
                           'Cost-burdened (30%) with at least one extra bedroom (Lower)': hh2p60o_2r_cb30l,
                           'Cost-burdened (30%) 60 with at least one extra bedroom (Upper)': hh2p60o_2r_cb30u,
                           'Cost-burdened (30%) with at least two extra bedrooms': hh2p60o_3r_cb30,
                           'Cost-burdened (30%) with at least two extra bedrooms MoE': hh2p60o_3r_cb30moe,
                           'Cost-burdened (30%) with at least two extra bedrooms MoE (%)': hh2p60o_3r_cb30moep,
                           'Cost-burdened (30%) with at least two extra bedrooms (Lower)': hh2p60o_3r_cb30l,
                           'Cost-burdened (30%) 60 with at least two extra bedrooms (Upper)': hh2p60o_3r_cb30u,
                           'Cost-burdened (50%) with at least one extra bedroom': hh2p60o_2r_cb50,
                           'Cost-burdened (50%) with at least one extra bedroom MoE': hh2p60o_2r_cb50moe,
                           'Cost-burdened (50%) with at least one extra bedroom MoE (%)': hh2p60o_2r_cb50moep,
                           'Cost-burdened (50%) with at least one extra bedroom (Lower)': hh2p60o_2r_cb50l,
                           'Cost-burdened (50%) with at least one extra bedroom (Upper)': hh2p60o_2r_cb50u,
                           'Cost-burdened (50%) with at least two extra bedrooms': hh2p60o_3r_cb50,
                           'Cost-burdened (50%) with at least two extra bedrooms MoE': hh2p60o_3r_cb50moe,
                           'Cost-burdened (50%) with at least two extra bedrooms MoE (%)': hh2p60o_3r_cb50moep,
                           'Cost-burdened (50%) with at least two extra bedrooms (Lower)': hh2p60o_3r_cb50l,
                           'Cost-burdened (50%) with at least two extra bedrooms (Upper)': hh2p60o_3r_cb50u}

supplypuma_60plus = {'PUMA': PUMAs_study,
                     'PUMA Name': pumanames,
                     'All occupied housing units': hhs_all,
                     'All occupied housing units MoE': hhs_allmoe,
                     'All occupied housing units MoE (%)': hhs_allmoep,
                     'All occupied housing units (Lower)': hhs_alll,
                     'All occupied housing units (Upper)': hhs_allu,                            
                     'At least one extra bedroom': hh12p60o_2r,
                     'At least one extra bedroom MoE': hh12p60o_2rmoe,
                     'At least one extra bedroom MoE (%)': hh12p60o_2rmoep,
                     'At least one extra bedroom (Lower)': hh12p60o_2rl,
                     'At least one extra bedroom (Upper)': hh12p60o_2ru,
                     'At least two extra bedrooms': hh12p60o_3r,
                     'At least two extra bedrooms MoE': hh12p60o_3rmoe,
                     'At least two extra bedrooms MoE (%)': hh12p60o_3rmoep,
                     'At least two extra bedrooms (Lower)': hh12p60o_3rl,
                     'At least two extra bedrooms (Upper)': hh12p60o_3ru,
                     'Cost-burdened (30%) with at least one extra bedroom': hh12p60o_2r_cb30,
                     'Cost-burdened (30%) with at least one extra bedroom MoE': hh12p60o_2r_cb30moe,
                     'Cost-burdened (30%) with at least one extra bedroom MoE (%)': hh12p60o_2r_cb30moep,
                     'Cost-burdened (30%) with at least one extra bedroom (Lower)': hh12p60o_2r_cb30l,
                     'Cost-burdened (30%) 60 with at least one extra bedroom (Upper)': hh12p60o_2r_cb30u,
                     'Cost-burdened (30%) with at least two extra bedrooms': hh12p60o_3r_cb30,
                     'Cost-burdened (30%) with at least two extra bedrooms MoE': hh12p60o_3r_cb30moe,
                     'Cost-burdened (30%) with at least two extra bedrooms MoE (%)': hh12p60o_3r_cb30moep,
                     'Cost-burdened (30%) with at least two extra bedrooms (Lower)': hh12p60o_3r_cb30l,
                     'Cost-burdened (30%) 60 with at least two extra bedrooms (Upper)': hh12p60o_3r_cb30u,
                     'Cost-burdened (50%) with at least one extra bedroom': hh12p60o_2r_cb50,
                     'Cost-burdened (50%) with at least one extra bedroom MoE': hh12p60o_2r_cb50moe,
                     'Cost-burdened (50%) with at least one extra bedroom MoE (%)': hh12p60o_2r_cb50moep,
                     'Cost-burdened (50%) with at least one extra bedroom (Lower)': hh12p60o_2r_cb50l,
                     'Cost-burdened (50%) with at least one extra bedroom (Upper)': hh12p60o_2r_cb50u,
                     'Cost-burdened (50%) with at least two extra bedrooms': hh12p60o_3r_cb50,
                     'Cost-burdened (50%) with at least two extra bedrooms MoE': hh12p60o_3r_cb50moe,
                     'Cost-burdened (50%) with at least two extra bedrooms MoE (%)': hh12p60o_3r_cb50moep,
                     'Cost-burdened (50%) with at least two extra bedrooms (Lower)': hh12p60o_3r_cb50l,
                     'Cost-burdened (50%) with at least two extra bedrooms (Upper)': hh12p60o_3r_cb50u}
                              
supplypuma_single65plus = {'PUMA': PUMAs_study,
                           'PUMA Name': pumanames,
                           'All occupied housing units': hhs_all,
                           'All occupied housing units MoE': hhs_allmoe,
                           'All occupied housing units MoE (%)': hhs_allmoep,
                           'All occupied housing units (Lower)': hhs_alll,
                           'All occupied housing units (Upper)': hhs_allu,
                           'At least one extra bedroom': hh1p65o_2r,
                           'At least one extra bedroom MoE': hh1p65o_2rmoe,
                           'At least one extra bedroom MoE (%)': hh1p65o_2rmoep,
                           'At least one extra bedroom (Lower)': hh1p65o_2rl,
                           'At least one extra bedroom (Upper)': hh1p65o_2ru,
                           'At least two extra bedrooms': hh1p65o_3r,
                           'At least two extra bedrooms MoE': hh1p65o_3rmoe,
                           'At least two extra bedrooms MoE (%)': hh1p65o_3rmoep,
                           'At least two extra bedrooms (Lower)': hh1p65o_3rl,
                           'At least two extra bedrooms (Upper)': hh1p65o_3ru,
                           'Cost-burdened (30%) with at least one extra bedroom': hh1p65o_2r_cb30,
                           'Cost-burdened (30%) with at least one extra bedroom MoE': hh1p65o_2r_cb30moe,
                           'Cost-burdened (30%) with at least one extra bedroom MoE (%)': hh1p65o_2r_cb30moep,
                           'Cost-burdened (30%) with at least one extra bedroom (Lower)': hh1p65o_2r_cb30l,
                           'Cost-burdened (30%) with at least one extra bedroom (Upper)': hh1p65o_2r_cb30u,
                           'Cost-burdened (30%) with at least two extra bedrooms': hh1p65o_3r_cb30,
                           'Cost-burdened (30%) with at least two extra bedrooms MoE': hh1p65o_3r_cb30moe,
                           'Cost-burdened (30%) with at least two extra bedrooms MoE (%)': hh1p65o_3r_cb30moep,
                           'Cost-burdened (30%) with at least two extra bedrooms (Lower)': hh1p65o_3r_cb30l,
                           'Cost-burdened (30%) with at least two extra bedrooms (Upper)': hh1p65o_3r_cb30u,
                           'Cost-burdened (50%) with at least one extra bedroom': hh1p65o_2r_cb50,
                           'Cost-burdened (50%) with at least one extra bedroom MoE': hh1p65o_2r_cb50moe,
                           'Cost-burdened (50%) with at least one extra bedroom MoE (%)': hh1p65o_2r_cb50moep,
                           'Cost-burdened (50%) with at least one extra bedroom (Lower)': hh1p65o_2r_cb50l,
                           'Cost-burdened (50%) with at least one extra bedroom (Upper)': hh1p65o_2r_cb50u,
                           'Cost-burdened (50%) with at least two extra bedrooms': hh1p65o_3r_cb50,
                           'Cost-burdened (50%) with at least two extra bedrooms MoE': hh1p65o_3r_cb50moe,
                           'Cost-burdened (50%) with at least two extra bedrooms MoE (%)': hh1p65o_3r_cb50moep,
                           'Cost-burdened (50%) with at least two extra bedrooms (Lower)': hh1p65o_3r_cb50l,
                           'Cost-burdened (50%) with at least two extra bedrooms (Upper)': hh1p65o_3r_cb50u}
                           
supplypuma_couple65plus = {'PUMA': PUMAs_study,
                           'PUMA Name': pumanames,
                           'All occupied housing units': hhs_all,
                           'All occupied housing units MoE': hhs_allmoe,
                           'All occupied housing units MoE (%)': hhs_allmoep,
                           'All occupied housing units (Lower)': hhs_alll,
                           'All occupied housing units (Upper)': hhs_allu,
                           'At least one extra bedroom': hh2p65o_2r,
                           'At least one extra bedroom MoE': hh2p65o_2rmoe,
                           'At least one extra bedroom MoE (%)': hh2p65o_2rmoep,
                           'At least one extra bedroom (Lower)': hh2p65o_2rl,
                           'At least one extra bedroom (Upper)': hh2p65o_2ru,
                           'At least two extra bedrooms': hh2p65o_3r,
                           'At least two extra bedrooms MoE': hh2p65o_3rmoe,
                           'At least two extra bedrooms MoE (%)': hh2p65o_3rmoep,
                           'At least two extra bedroom (Lower)': hh2p65o_3rl,
                           'At least two extra bedroom (Upper)': hh2p65o_3ru,
                           'Cost-burdened (30%) with at least one extra bedroom': hh2p65o_2r_cb30,
                           'Cost-burdened (30%) with at least one extra bedroom MoE': hh2p65o_2r_cb30moe,
                           'Cost-burdened (30%) with at least one extra bedroom MoE (%)': hh2p65o_2r_cb30moep,
                           'Cost-burdened (30%) with at least one extra bedroom (Lower)': hh2p65o_2r_cb30l,
                           'Cost-burdened (30%) with at least one extra bedroom (Upper)': hh2p65o_2r_cb30u,
                           'Cost-burdened (30%) with at least two extra bedrooms': hh2p65o_3r_cb30,
                           'Cost-burdened (30%) with at least two extra bedrooms MoE': hh2p65o_3r_cb30moe,
                           'Cost-burdened (30%) with at least two extra bedrooms MoE (%)': hh2p65o_3r_cb30moep,
                           'Cost-burdened (30%) with at least two extra bedrooms (Lower)': hh2p65o_3r_cb30l,
                           'Cost-burdened (30%) with at least two extra bedrooms (Upper)': hh2p65o_3r_cb30u,
                           'Cost-burdened (50%) with at least one extra bedroom': hh2p65o_2r_cb50,
                           'Cost-burdened (50%) with at least one extra bedroom MoE': hh2p65o_2r_cb50moe,
                           'Cost-burdened (50%) with at least one extra bedroom MoE (%)': hh2p65o_2r_cb50moep,
                           'Cost-burdened (50%) with at least one extra bedroom (Lower)': hh2p65o_2r_cb50l,
                           'Cost-burdened (50%) with at least one extra bedroom (Upper)': hh2p65o_2r_cb50u,
                           'Cost-burdened (50%) with at least two extra bedrooms': hh2p65o_3r_cb50,
                           'Cost-burdened (50%) with at least two extra bedrooms MoE': hh2p65o_3r_cb50moe,
                           'Cost-burdened (50%) with at least two extra bedrooms MoE (%)': hh2p65o_3r_cb50moep,
                           'Cost-burdened (50%) with at least two extra bedrooms (Lower)': hh2p65o_3r_cb50l,
                           'Cost-burdened (50%) with at least two extra bedrooms (Upper)': hh2p65o_3r_cb50u}

supplypuma_65plus = {'PUMA': PUMAs_study,
                     'PUMA Name': pumanames,
                     'All occupied housing units': hhs_all,
                     'All occupied housing units MoE': hhs_allmoe,
                     'All occupied housing units MoE (%)': hhs_allmoep,
                     'All occupied housing units (Lower)': hhs_alll,
                     'All occupied housing units (Upper)': hhs_allu,
                     'At least one extra bedroom': hh12p65o_2r,
                     'At least one extra bedroom MoE': hh12p65o_2rmoe,
                     'At least one extra bedroom MoE (%)': hh12p65o_2rmoep,
                     'At least one extra bedroom (Lower)': hh12p65o_2rl,
                     'At least one extra bedroom (Upper)': hh12p65o_2ru,
                     'At least two extra bedrooms': hh12p65o_3r,
                     'At least two extra bedrooms MoE': hh12p65o_3rmoe,
                     'At least two extra bedrooms MoE (%)': hh12p65o_3rmoep,
                     'At least two extra bedroom (Lower)': hh12p65o_3rl,
                     'At least two extra bedroom (Upper)': hh12p65o_3ru,
                     'Cost-burdened (30%) with at least one extra bedroom': hh12p65o_2r_cb30,
                     'Cost-burdened (30%) with at least one extra bedroom MoE': hh12p65o_2r_cb30moe,
                     'Cost-burdened (30%) with at least one extra bedroom MoE (%)': hh12p65o_2r_cb30moep,
                     'Cost-burdened (30%) with at least one extra bedroom (Lower)': hh12p65o_2r_cb30l,
                     'Cost-burdened (30%) with at least one extra bedroom (Upper)': hh12p65o_2r_cb30u,
                     'Cost-burdened (30%) with at least two extra bedrooms': hh12p65o_3r_cb30,
                     'Cost-burdened (30%) with at least two extra bedrooms MoE': hh12p65o_3r_cb30moe,
                     'Cost-burdened (30%) with at least two extra bedrooms MoE (%)': hh12p65o_3r_cb30moep,
                     'Cost-burdened (30%) with at least two extra bedrooms (Lower)': hh12p65o_3r_cb30l,
                     'Cost-burdened (30%) with at least two extra bedrooms (Upper)': hh12p65o_3r_cb30u,
                     'Cost-burdened (50%) with at least one extra bedroom': hh12p65o_2r_cb50,
                     'Cost-burdened (50%) with at least one extra bedroom MoE': hh12p65o_2r_cb50moe,
                     'Cost-burdened (50%) with at least one extra bedroom MoE (%)': hh12p65o_2r_cb50moep,
                     'Cost-burdened (50%) with at least one extra bedroom (Lower)': hh12p65o_2r_cb50l,
                     'Cost-burdened (50%) with at least one extra bedroom (Upper)': hh12p65o_2r_cb50u,
                     'Cost-burdened (50%) with at least two extra bedrooms': hh12p65o_3r_cb50,
                     'Cost-burdened (50%) with at least two extra bedrooms MoE': hh12p65o_3r_cb50moe,
                     'Cost-burdened (50%) with at least two extra bedrooms MoE (%)': hh12p65o_3r_cb50moep,
                     'Cost-burdened (50%) with at least two extra bedrooms (Lower)': hh12p65o_3r_cb50l,
                     'Cost-burdened (50%) with at least two extra bedrooms (Upper)': hh12p65o_3r_cb50u}

intergen_PUMA_single60plus = pd.DataFrame(data = supplypuma_single60plus)
filedest = "K:\\DataServices\\Projects\\Current_Projects\\Housing\\Intergenerational_Homesharing\\Data\\Tabular\\intergen_pumas_single_60plus.csv"
intergen_PUMA_single60plus.to_csv(filedest)

intergen_PUMA_couple60plus = pd.DataFrame(data = supplypuma_couple60plus)
filedest = "K:\\DataServices\\Projects\\Current_Projects\\Housing\\Intergenerational_Homesharing\\Data\\Tabular\\intergen_pumas_couple_60plus.csv"
intergen_PUMA_couple60plus.to_csv(filedest)

intergen_PUMA_60plus = pd.DataFrame(data = supplypuma_60plus)
filedest = "K:\\DataServices\\Projects\\Current_Projects\\Housing\\Intergenerational_Homesharing\\Data\\Tabular\\intergen_pumas_60plus.csv"
intergen_PUMA_couple60plus.to_csv(filedest)


intergen_PUMA_single65plus = pd.DataFrame(data = supplypuma_single65plus)
filedest = "K:\\DataServices\\Projects\\Current_Projects\\Housing\\Intergenerational_Homesharing\\Data\\Tabular\\intergen_pumas_single_65plus.csv"
intergen_PUMA_single65plus.to_csv(filedest)

intergen_PUMA_couple65plus = pd.DataFrame(data = supplypuma_couple65plus)
filedest = "K:\\DataServices\\Projects\\Current_Projects\\Housing\\Intergenerational_Homesharing\\Data\\Tabular\\intergen_pumas_couple_65plus.csv"
intergen_PUMA_couple65plus.to_csv(filedest)

intergen_PUMA_65plus = pd.DataFrame(data = supplypuma_65plus)
filedest = "K:\\DataServices\\Projects\\Current_Projects\\Housing\\Intergenerational_Homesharing\\Data\\Tabular\\intergen_pumas_65plus.csv"
intergen_PUMA_65plus.to_csv(filedest)

# # PUMAs
# def errplot(x, y, yerr, **kwargs):
#     ax = plt.gca()
#     data = kwargs.pop("data")
#     data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)
    
# sns.set(style = 'whitegrid')
# ax = sns.barplot(x='PUMA', y="Immigrant Entrepreneurs Pct", yerr = ImmEnt_PUMA["Immigrant Entrepreneurs Pct MoE"]*1, data = ImmEnt_PUMA)



