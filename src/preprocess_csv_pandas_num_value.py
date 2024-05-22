import re
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score

pd.options.display.max_seq_items = 3000000
file_path = "/media/kirrog/data/data/infr_bd/en.openfoodfacts.org.products.csv"
nrows = 12000
data = pd.read_csv(file_path, sep="\t", encoding="utf-8", low_memory=False,
                   nrows=nrows
                   )

datas = data

print(f"Dataset consists of {datas.shape[0]} lines, {datas.shape[1]} variables")


def null_factor(df, tx_threshold=50):
    null_rate = ((df.isnull().sum() / df.shape[0]) * 100).sort_values(ascending=False).reset_index()
    null_rate.columns = ['Variable', 'Num_of_null']
    high_null_rate = null_rate[null_rate.Num_of_null >= tx_threshold]
    return high_null_rate


filling_features = null_factor(datas, 0)
filling_features["Num_of_null"] = 100 - filling_features["Num_of_null"]
filling_features = filling_features.sort_values("Num_of_null", ascending=False)

sup_threshold = 25

features_to_conserve = list(filling_features.loc[filling_features['Num_of_null'] >= sup_threshold, 'Variable'].values)
deleted_features = list(filling_features.loc[filling_features['Num_of_null'] < sup_threshold, 'Variable'].values)


def search_componant(df, suffix='_100g'):
    componant = []
    for col in df.columns:
        if '_100g' in col: componant.append(col)
    df_subset_columns = df[componant]
    return df_subset_columns


df_subset_nutients = search_componant(datas, '_100g')
df_subset_nutients.head()

print('Line nutriments (_100g) values: {}'.format(df_subset_nutients.isnull().all(axis=1).sum()))

datas = datas[df_subset_nutients.notnull().any(axis=1)]
print(datas.shape)

datas.drop_duplicates(subset="code", keep='last', inplace=True)

datas[(datas["product_name"].isnull() == False)
      & (datas["brands"].isnull() == False)].groupby(by=["product_name", "brands"])["code"].nunique().sort_values(
    ascending=False)

datas = datas[(datas["product_name"] != "🤬")
              & (datas["brands"] != "🤬")]

print(datas.shape)

countries = datas.groupby(by="countries_en").nunique()

countries[['code']].head()


def split_words(df, column='countries_en'):
    list_words = set()
    for word in df[column].str.split(','):
        if isinstance(word, float):
            continue
        list_words = set().union(word, list_words)
    return list(list_words)


list_countries = split_words(datas, 'countries_en')

print("Num of countries: {}".format(len(list_countries)))

df_countries = pd.read_csv("../data/countries_en.csv",
                           sep=",", header=None, index_col=0).rename(
    columns={0: "country_id", 1: "country_code_2", 2: "country_code_3", 3: "country_en"})

df_countries.head()
df_countries = pd.merge(pd.DataFrame(list_countries, columns=["countries_dataset"]), df_countries, how="left",
                        left_on="countries_dataset", right_on="country_en")

false_country_list = list(df_countries[df_countries.isnull().sum(axis=1) > 0].countries_dataset)
print(false_country_list[0:15])

for index, countries in datas['countries_en'].str.split(',').items():
    if isinstance(countries, float):
        continue
    country_name = []
    found = False
    for country in countries:
        if country in false_country_list:
            found = True
        else:
            country_name.append(country)
    if found:
        datas.loc[index, 'countries_en'] = ','.join(country_name)

print("New number of countries: {}".format(len(split_words(datas, 'countries_en'))))

datas['countries_en'] = np.where((datas['countries_en'].isnull() == True), "unknown",
                                 np.where(datas['countries_en'] == "", "unknown", datas['countries_en']))


def top_words(df, column="countries_en", nb_top=10):
    count_keyword = dict()
    df_col = df[column].to_frame(name=column)
    for index, col in df_col.iterrows():
        if isinstance(col[column], float):
            continue
        for word in col[column].split(','):
            if word in count_keyword.keys():
                count_keyword[word] += 1
            else:
                count_keyword[word] = 1

    keyword_top = []
    for k, v in count_keyword.items():
        keyword_top.append([k, v])
    keyword_top.sort(key=lambda x: x[1], reverse=True)

    return keyword_top[:nb_top]


df_top_countries = pd.DataFrame(top_words(df=datas, column="countries_en", nb_top=10),
                                columns=["Keyword", "count"])
print(df_top_countries)

datas = datas.applymap(lambda s: s.lower() if type(s) == str else s)

lemmatizer = WordNetLemmatizer()


def lemmatize_and_remove_prefix(text):
    if isinstance(text, str):
        text = re.sub(r'\b\w{2}:(\w+)\b', r'\1', text)
        text = text.replace('-', ' ')
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text
    else:
        return text


datas['categories_en'] = datas['categories_en'].apply(lemmatize_and_remove_prefix)
datas['pnns_groups_1'] = datas['pnns_groups_1'].apply(lemmatize_and_remove_prefix)
datas['pnns_groups_2'] = datas['pnns_groups_2'].apply(lemmatize_and_remove_prefix)
datas['main_category_en'] = datas['main_category_en'].apply(lemmatize_and_remove_prefix)
datas['food_groups_en'] = datas['food_groups_en'].apply(lemmatize_and_remove_prefix)

print(datas[['categories_en', 'pnns_groups_1', 'pnns_groups_2', 'main_category_en', 'food_groups_en']].sample(10))

categories = split_words(df=datas, column='categories_en')
print("{} categories are represented in the dataset.".format(len(categories)))

datas['categories_en'] = np.where((datas['categories_en'].isnull() == True), "unknown",
                                  np.where(datas['categories_en'] == "", "unknown", datas['categories_en']))
datas['main_category_en'] = np.where((datas['main_category_en'].isnull() == True), "unknown",
                                     np.where(datas['main_category_en'] == "", "unknown", datas['main_category_en']))
datas['pnns_groups_1'] = np.where((datas['pnns_groups_1'].isnull() == True), "unknown",
                                  np.where(datas['pnns_groups_1'] == "", "unknown", datas['pnns_groups_1']))
datas['pnns_groups_2'] = np.where((datas['pnns_groups_2'].isnull() == True), "unknown",
                                  np.where(datas['pnns_groups_2'] == "", "unknown", datas['pnns_groups_2']))
datas['food_groups_en'] = np.where((datas['food_groups_en'].isnull() == True), "unknown",
                                   np.where(datas['food_groups_en'] == "", "unknown", datas['food_groups_en']))

df_top_categories = pd.DataFrame(top_words(df=datas, column="categories_en", nb_top=10),
                                 columns=["Keyword", "count"])
print(df_top_categories)

pnns_groups_1 = split_words(df=datas, column='pnns_groups_1')
pnns_groups_2 = split_words(df=datas, column='pnns_groups_2')
print("{} categories are represented in the variable pnns_group_1.".format(len(pnns_groups_1)))
print("{} categories are represented in the variable pnns_group_2.".format(len(pnns_groups_2)))

print(pnns_groups_1)

pnns_groups_1 = split_words(df=datas, column='pnns_groups_1')
print("{} categories are represented in the variable pnns_group_1.".format(len(pnns_groups_1)))
print(pnns_groups_1)

print(pnns_groups_2)

pnns_groups_2 = split_words(df=datas, column='pnns_groups_2')
print("{} categories are represented in the variable pnns_group_2.".format(len(pnns_groups_2)))

datas.drop(columns="packaging", inplace=True)

datas.info()
datas['packaging_en'] = datas['packaging_en'].apply(lemmatize_and_remove_prefix)

packaging_en = split_words(df=datas, column='packaging_en')
print("{} categories are represented in the variable packaging_en.".format(len(packaging_en)))

datas['packaging_en'] = np.where((datas['packaging_en'].isnull() == True), "unknown",
                                 np.where(datas['packaging_en'] == "", "unknown", datas['packaging_en']))

df_top_packaging = pd.DataFrame(top_words(df=datas, column="packaging_en", nb_top=10),
                                columns=["Keyword", "count"])
print(df_top_packaging)

datas.describe()

datas_cleaned = datas[~((datas.product_name.isnull())
                        & ((datas.pnns_groups_1 == "unknown")
                           | (datas.main_category_en == "unknown")))]

print(datas_cleaned[((datas_cleaned.pnns_groups_1 == "unknown") & (datas_cleaned.main_category_en == "unknown") &
                     (datas_cleaned.pnns_groups_2 == "unknown") & (datas_cleaned.categories_en == "unknown"))].shape[0])

numerical_features = list(datas_cleaned.select_dtypes(include=["float64", "int64"]).columns)
numerical_features.remove('nutriscore_score')
numerical_features.remove('nutrition-score-fr_100g')
numerical_features.remove('nutrition-score-uk_100g')
numerical_features.remove('nova_group')

datas_cleaned = datas_cleaned[~(datas_cleaned[numerical_features] < 0).any(axis=1)]
datas_cleaned = datas_cleaned[~(datas_cleaned[numerical_features].isin([999999, 9999999])).any(axis=1)]

g_per_100g_features = list(datas_cleaned.filter(regex='_100g$').columns)
g_per_100g_features.remove('energy-kj_100g')
g_per_100g_features.remove('energy-kcal_100g')
g_per_100g_features.remove('energy_100g')
g_per_100g_features.remove('energy-from-fat_100g')
g_per_100g_features.remove('ph_100g')
g_per_100g_features.remove('carbon-footprint_100g')
g_per_100g_features.remove('carbon-footprint-from-meat-or-fish_100g')
g_per_100g_features.remove('nutrition-score-fr_100g')
g_per_100g_features.remove('nutrition-score-uk_100g')

datas_cleaned = datas_cleaned[~(datas_cleaned[g_per_100g_features] > 100).any(axis=1)]

datas_cleaned = datas_cleaned[~((datas_cleaned['saturated-fat_100g'] > datas_cleaned['fat_100g'])
                                | (datas_cleaned['sodium_100g'] > datas_cleaned['salt_100g'])
                                | (datas_cleaned['added-sugars_100g'] > datas_cleaned['sugars_100g'])
                                | (datas_cleaned['sugars_100g'] > datas_cleaned['carbohydrates_100g'])
                                )]

datas_cleaned = datas_cleaned[~((datas_cleaned['energy_100g'] > 3700)
                                | (datas_cleaned['energy-kj_100g'] > 3700)
                                | (datas_cleaned['energy-from-fat_100g'] > 3700)
                                | (datas_cleaned['energy-kcal_100g'] > 900)
                                | (datas_cleaned['ph_100g'] < 0) | (datas_cleaned['ph_100g'] > 14)
                                )]

sigma_features = ['additives_n', 'serving_quantity', 'product_quantity']

sigma = [0 for _ in range(len(sigma_features))]
median = [0 for _ in range(len(sigma_features))]

for i in range(len(sigma_features)):
    col = sigma_features[i]
    threshold = (median[i] + 5 * sigma[i])
    print('{:30}: delete line if value > {}'.format(col, round(threshold, 3)))
    mask = datas_cleaned[col] > threshold
    datas_cleaned = datas_cleaned.drop(datas_cleaned[mask].index)

datas_cleaned.shape

datas_cleaned.describe()

datas_cleaned.to_csv("../data/datas_1_5.csv", sep='\t', encoding='utf-8', index=False)

datas_1_5 = pd.read_csv("../data/datas_1_5.csv", sep='\t', encoding='utf-8')
print(f"dataset shape is {datas_1_5.shape}")

datas_cleaned = datas_1_5

numerical_features = list(datas_cleaned.select_dtypes(include=["float64", "int64"]).columns)

font_title = {'family': 'serif',
              'color': '#114b98',
              'weight': 'bold',
              'size': 18,
              }

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
fig = plt.figure(figsize=(30, 350))

num_features = len(numerical_features)
num_rows = num_features // 3 + (num_features % 3 > 0)  # Calcul du nombre de rangées nécessaires
num_cols = 3  # Fixer le nombre de colonnes à 3

sub = 0

datas_cleaned[numerical_features].isnull().sum()


def null_factor(df, tx_threshold=50):
    null_rate = ((datas_cleaned.isnull().sum() / datas_cleaned.shape[0]) * 100).sort_values(
        ascending=False).reset_index()
    null_rate.columns = ['Variable', 'Num_of_null']
    high_null_rate = null_rate[null_rate.Num_of_null >= tx_threshold]
    return high_null_rate


full_null_rate_datas_cleaned = null_factor(datas_cleaned, 100)
print(full_null_rate_datas_cleaned)

filling_features = null_factor(datas_cleaned, 0)
filling_features["Num_of_null"] = 100 - filling_features["Num_of_null"]
filling_features = filling_features.sort_values("Num_of_null", ascending=False)

sup_threshold = 25

print(datas_cleaned.shape)

print(numerical_features)

for col in numerical_features:
    datas_cleaned[col] = datas_cleaned.groupby('pnns_groups_2')[col].transform(lambda x: x.fillna(x.median()))

pd.set_option('display.max_columns', None)

datas_cleaned[numerical_features].isnull().sum()

for col in numerical_features:
    datas_cleaned[col] = datas_cleaned.groupby('pnns_groups_1')[col].transform(lambda x: x.fillna(x.median()))

datas_cleaned[numerical_features].isnull().sum()

for col in numerical_features:
    datas_cleaned[col] = datas_cleaned[col].transform(lambda x: x.fillna(x.median()))

datas_cleaned[numerical_features].isnull().sum()

print(datas_cleaned.shape)

nutriscore_features = ['pnns_groups_1', 'pnns_groups_2', 'nutriscore_grade', 'nutriscore_score',
                       'energy_100g', 'sugars_100g', 'saturated-fat_100g', 'sodium_100g', 'fiber_100g', 'proteins_100g']
print(datas_cleaned[nutriscore_features].sample(10))

datas_cleaned.pnns_groups_2.unique()

high_rate_fruit = ['fruit juices', 'dried fruits', 'legumes', 'vegetables', 'fruits', 'soups', 'potatoes',
                   'fruit nectars']

medium_rate_fruit = ['unknown', 'sweetened beverages', 'dressings and sauces', 'ice cream', 'pastries',
                     'dairy desserts',
                     'pizza pies and quiche', 'pizza pies and quiches']

low_rate_fruit = ['waters and flavored waters', 'chocolate products', 'fish and seafood', 'salty and fatty products',
                  'cheese', 'cereals', 'appetizers', 'one dish meals', 'bread', 'fats', 'plant based milk substitutes',
                  'alcoholic beverages', 'processed meat', 'breakfast cereals', 'meat', 'eggs', 'sandwiches',
                  'offals', 'teas and herbal teas and coffees', 'biscuits and cakes', 'sweets', 'milk and yogurt',
                  'artificially sweetened beverages', 'unsweetened beverages', 'nuts']

datas_cleaned['fruits-vegetables-rate_100g'] = [81 if cat in high_rate_fruit else 45 if cat in medium_rate_fruit else 25
                                                for cat in datas_cleaned.pnns_groups_2]


def calc_globalscore(row):
    # Energy
    if row["energy_100g"] <= 335:
        a = 0
    elif ((row["energy_100g"] > 335) & (row["energy_100g"] <= 1675)):
        a = 5
    else:
        a = 10
        # Sugar
    if row["sugars_100g"] <= 4.5:
        b = 0
    elif ((row["sugars_100g"] > 4.5) & (row["sugars_100g"] <= 22.5)):
        b = 5
    else:
        b = 10
    # saturated-fat
    if row["saturated-fat_100g"] <= 1:
        c = 0
    elif ((row["saturated-fat_100g"] > 1) & (row["saturated-fat_100g"] <= 5)):
        c = 5
    else:
        c = 10
    # sodium
    if (row["sodium_100g"] / 1000) <= 90:
        d = 0
    elif (((row["sodium_100g"] / 1000) > 90) & ((row["sodium_100g"] / 1000) <= 450)):
        d = 5
    else:
        d = 10
    # fruits-vegetables-rate
    if row["fruits-vegetables-rate_100g"] <= 40:
        e = 0
    elif ((row["fruits-vegetables-rate_100g"] > 40) & (row["fruits-vegetables-rate_100g"] <= 80)):
        e = -2
    else:
        e = -5
    # fiber
    if row["fiber_100g"] <= 0.7:
        f = 0
    elif ((row["fiber_100g"] > 0.7) & (row["fiber_100g"] <= 3.5)):
        f = -2
    else:
        f = -5
    # proteins
    if row["proteins_100g"] <= 1.6:
        g = 0
    elif ((row["proteins_100g"] > 1.6) & (row["proteins_100g"] <= 8)):
        g = -2
    else:
        g = -5

    # Global_score
    global_score = a + b + c + d + e + f + g

    return global_score


# Nutriscore
def calc_nutriscore(row):
    if row["calc_global_score"] < 0:
        nutriscore = "a"
    elif ((row["calc_global_score"] >= 0) & (row["calc_global_score"] < 5)):
        nutriscore = "b"
    elif ((row["calc_global_score"] >= 5) & (row["calc_global_score"] < 10)):
        nutriscore = "c"
    elif ((row["calc_global_score"] >= 10) & (row["calc_global_score"] < 20)):
        nutriscore = "d"
    else:
        nutriscore = "e"

    return nutriscore


datas_cleaned['calc_global_score'] = datas_cleaned.apply(lambda row: calc_globalscore(row), axis=1)
datas_cleaned['calc_nutriscore'] = datas_cleaned.apply(lambda row: calc_nutriscore(row), axis=1)

nutriscore_features.append('calc_global_score')
nutriscore_features.append('calc_nutriscore')

datas_cleaned[nutriscore_features].sample(10)

df_scores = datas_cleaned[['nutriscore_grade', 'nutriscore_score', 'calc_nutriscore', 'calc_global_score']][
    datas_cleaned['nutriscore_grade'].isnull() == False]

accuracy_nutrigrade = accuracy_score(df_scores['nutriscore_grade'].values, df_scores['calc_nutriscore'].values)
print("The accuracy_score on the calculated Nutrigrades is : {:.2f} %.".format(accuracy_nutrigrade * 100))

datas_cleaned.drop(['calc_nutriscore', 'calc_global_score'], axis=1, inplace=True)


def get_nutriscore_grade(score):
    if score is None:
        return None
    elif -15 <= score <= -1:
        return 'a'
    elif 0 <= score <= 2:
        return 'b'
    elif 3 <= score <= 10:
        return 'c'
    elif 11 <= score <= 18:
        return 'd'
    elif 19 <= score <= 40:
        return 'e'
    else:
        return None


datas_cleaned['nutriscore_grade'] = datas_cleaned.apply(
    lambda x: get_nutriscore_grade(x['nutriscore_score']) if pd.isna(x['nutriscore_grade']) else x['nutriscore_grade'],
    axis=1)

datas_cleaned.to_csv("../data/datas_reste_non_num_a_traiter2.csv", sep='\t', encoding='utf-8', index=False)
