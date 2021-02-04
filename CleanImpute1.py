import pandas as pd
import seaborn as sns
import numpy as np
import tools
import math as m

# Zeige immer alle Spalten eines DataFrames an
pd.set_option('display.max_columns', None)

df = pd.read_csv('immo_data_clean.csv', index_col=0)

#############
# totalRent #
#############

# Warmmiete: Für unsere Prognose nicht relevant, da nicht einheitlich ermittelt

df.drop('totalRent', axis=1, inplace=True)

###########
# Outlier #
###########

# Identifiziere Outlier

rent_upper_bound = 1.5e+03
rent_lower_bound = 100
#outlier = df.loc[(df.baseRent > rent_upper_bound)|(df.baseRent < rent_lower_bound)]
#outlier.shape

# Drop Outlier (nur ~13600 Einträge über 1500€ oder unter 100€ Kaltmiete)

df_cut = df.loc[(df.baseRent < rent_upper_bound) & (df.baseRent > rent_lower_bound)]

##############
# typeOfFlat #
##############

#df['typeOfFlat'].unique()
#sns.violinplot(x='typeOfFlat', y='baseRent', data=df_cut)

# Zwei Kategorien zu erkennen:
# 1. Kategorie: teure Types (vgl. exp_typeOfFlat unten)
# 2. Ketegorie: normale Types 

exp_typeOfFlat = ['penthouse', 'loft', 'maisonette', 'terraced_flat']
df_cut.loc[df_cut.typeOfFlat.isin(exp_typeOfFlat)].shape

# Nur 13637 Appartments sind vom teuren Type
# ---> Reduktion der Analyse auf normale Types 

df_cut_flat = df_cut.loc[~df_cut.typeOfFlat.isin(exp_typeOfFlat)]
#sns.violinplot(x='typeOfFlat', y='baseRent', data=df_cut_flat)

# Die restlichen Typen weisen eine sehr ähnliche Verteilung der Kaltmiete auf
# ---> Spalte droppen

df_flat_drop = df_cut_flat.drop('typeOfFlat', axis = 1)
df_1 = df_flat_drop


#########################
# yearConstructed/Range #
#########################

#sns.violinplot(x='yearConstructedRange', y='baseRent', data=df_1)

# Kategorien erklärt:

#df_1.loc[df_1.yearConstructedRange.isin([1,2,3,4,5])].yearConstructed.describe()

# yearConstructedRange = 1: Bau von ....-1950
# yearConstructedRange = 2: Bau von 1951-1970
# yearConstructedRange = 3: Bau von 1971-1980
# yearConstructedRange = 4: Bau von 1981-1990
# yearConstructedRange = 5: Bau von 1991-2000
# yearConstructedRange = 6: Bau von 2001-2005
# yearConstructedRange = 7: Bau von 2006-2010
# yearConstructedRange = 8: Bau von 2011-2015
# yearConstructedRange = 9: Bau von 2016-....

# Plot legt nahe, Kategorien 1-5 von yearConstructedRange zusammenzufassen

df_1['yearConstructedRange_new'] = df_1['yearConstructedRange'].map(lambda x: 1 if x<=5 else x-4)

# Impute fehlende Werte nach empirischer Verteilung der vorhandenen Werte
df_1 = tools.categorical_dist_imputer(df_1, 'yearConstructedRange_new')
# Droppe yearConstructedRange, die durch yearConstructedRangeNew ersetzt wurde
df_1.drop('yearConstructedRange', axis=1, inplace=True)

# Droppe yearConstructed, da die Wesentliche Information nun vereinfacht in
# yearConstructedRange_new drin steckt
df_1.drop('yearConstructed', axis=1, inplace=True)

########################
# floor/numberOfFloors #
########################

#sns.violinplot(x='floor', y='baseRent', data=df_1[df_1.floor < 20])
#df_1.loc[df_1.floor >5].shape

#sns.violinplot(x='numberOfFloors', y='baseRent', data=df_1[df_1.numberOfFloors < 20])
#df_1.loc[df_1.numberOfFloors >6].shape

# df_1.loc[df_1.floor.isnull() & ~df_1.numberOfFloors.isnull()].shape

# floor und numberOfFloors scheinen ungefähr denselben Einfluss auf baseRent zu haben
# Außerdem: floor NaN Wert => numberOfFloors NaN Wert
# ---> Wir behalten floors, und droppen numberOfFloors
# ---> fassen floors > 5 in eine Kategorie zusammen, da sich ab dort keine 
#      Regelmäßigkeit mehr abzeichnet  

df_1.drop('numberOfFloors', axis=1, inplace=True)
df_1['floor_new'] = df_1.floor.map(lambda x: 6 if x>5 else x)
df_1.drop('floor', axis=1, inplace=True)

# Impute fehlende Werte nach empirischer Verteilung der vorhandenen Werte
df_1 = tools.categorical_dist_imputer(df_1,'floor_new')

################
# InteriorQual #
################

df_1.loc[df['interiorQual'].isnull(),'interiorQual'] = 'nicht vorhanden'

#df_1.groupby('interiorQual').size()
#sns.violinplot(x='interiorQual', y='baseRent', data=df_1)

# Der Plot legt nahe, die Daten auf zwei Kategorien zu reduzieren:
# 0. Kategorie: normal, simple, nicht vorhanden
# 1. Kategorie: sophisticated, luxury
# (insbesondere werden dabei alle NaN values durch 'normal' ersetzt)

interiorQual_dict = {'normal':0, 'simple':0, 'nicht vorhanden':0, 'sophisticated':1, 'luxury':1}
df_1['interiorQual_new'] = df_1.interiorQual.map(interiorQual_dict)
df_1.drop('interiorQual', axis=1, inplace=True)


###############
# thermalChar #
###############

# Plotte den Mittelwert der Kaltmiete für jeden Abschnitt von thermalChar der Breite binwidth 
# Dazu: thermalChar diskretisieren, Mittelwerte über Gruppen bilden
# binwidth = 10
# df_1['thermalChar_discrete'] = df_1.thermalChar.apply(lambda x: x - m.fmod(x,binwidth))
# df_2 = df_1[['thermalChar_discrete','baseRent']].groupby('thermalChar_discrete', as_index=False).mean()
# sns.scatterplot(x='thermalChar_discrete', y='baseRent', data=df_2.loc[df_2['thermalChar_discrete']<550])    

# Der einzige Bereich, der einen Zusammenhang erklären könnte ist der von thermalChar in (80,150)
# sns.violinplot(x='yearConstructedRange_new', y='thermalChar', data=df_1.loc[df_1.thermalChar < 550])
# df_1[['thermalChar', 'yearConstructedRange_new']].corr()
# sns.violinplot(x='noRooms', y='thermalChar', data=df_1.loc[df_1.thermalChar < 550])
# df_1[['thermalChar', 'noRooms']].corr()
# sns.histplot(x='baseRent', y='thermalChar', bins=30, data=df_1.loc[(df_1.baseRent<800) & (df_1.thermalChar <250)])
# sns.regplot(y='baseRent', x='thermalChar', data=df_1.loc[(df_1.baseRent<800) & (df_1.thermalChar <500)], scatter=False)

df_1.loc[(df_1['thermalChar']>80) &(df_1['thermalChar']<150)].shape
# Nur 81000 Daten liegen in einem Bereich, der einen Zusammenhang zur Kaltmiete vermuten lässt
# Da das nur 1/3 der Daten sind: Spalte droppen

df_1.drop('thermalChar', axis=1, inplace=True)


###########
# noRooms #
###########

# Drop Outlier 
noRooms_max = 7
df_1 = df_1.loc[df_1.noRooms <= noRooms_max]

# Runde die Raumzahlangaben auf 'halbe' Räume 
df_1['noRooms'] = df_1['noRooms'].multiply(2).round().multiply(1/2)

# sns.violinplot(x='noRooms', y='baseRent', data=df_1)

###############
# livingSpace #
###############

# Drop Outlier
livingSpace_max = 250
df_1 = df_1.loc[df_1.livingSpace <= livingSpace_max]

# sns.scatterplot(x='livingSpace', y='baseRent', data=df_1.loc[df_1.regio2 == 'Hamburg'])

#############
# condition #
#############

#sns.violinplot(y='baseRent', x='condition', data=df )

new = ['fully_renovated', 'first_time_use', 'modernized']
normal = ['well_kept', 'refurbished', 'mint_condition', 'first_time_use_after_refurbishment']
old = ['negotiable', 'need_of_renovation', 'ripe_for_demolition']

def condition_map(condition):
    if condition in new:
        return 1
    if condition in normal:
        return 2
    if condition in old:
        return 3

df_1['condition_new'] = df_1.condition.map(condition_map)
df_1 = tools.categorical_dist_imputer(df_1,'condition_new')

df_1.to_csv('data_imputed_1.csv')
