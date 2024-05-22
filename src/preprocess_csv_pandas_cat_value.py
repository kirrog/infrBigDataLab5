import numpy as np
import pandas as pd

pd.options.display.max_seq_items = 3000000
from sklearn.feature_extraction import FeatureHasher
from nltk.corpus import stopwords
import re
from sklearn import preprocessing
import nltk

nrows = 12000
file_path = "../data/datas_reste_non_num_a_traiter2.csv"
data = pd.read_csv(file_path, sep="\t", encoding="utf-8", low_memory=False,
                   nrows=nrows
                   )


def add_feature_hashed_column(df, column_name, n_features=1000):
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    hashed_column = hasher.transform(df[column_name].apply(lambda x: [x]))
    hashed_df = pd.DataFrame(hashed_column.toarray())
    hashed_df.columns = [column_name + '_' + str(i) for i in range(n_features)]
    df = pd.concat([df, hashed_df], axis=1)
    return df


li_col_del = list(data.columns[data.columns.str.contains('image')]) + \
             list(data.columns[data.columns.str.contains('tags')]) + \
             list(data.columns[data.columns.str.contains('states')])

data.drop(columns=li_col_del, inplace=True)

print(data.shape)

new_dtypes = {}

for col in data.columns.values:
    if data[col].dtype == 'object':
        if len(data[col].unique()) / len(data[col]) < 0.5:
            new_dtypes[col] = 'category'
        else:
            new_dtypes[col] = 'object'

# float64 -> float32
for col in data.columns.values:
    if data[col].dtype == 'float64':
        if data[col].notna().sum() != 0:
            new_dtypes[col] = 'float32'

# int64 -> int8
for col in data.columns.values:
    if data[col].dtype == 'int64':
        new_dtypes[col] = 'int8'

print(new_dtypes)

X_100g_cols = data.columns[data.columns.str.contains('_100g')]
for col in X_100g_cols:
    new_dtypes[col] = 'float32'

desc = data.describe(include='all')

for i, c in enumerate(data.columns):
    print('\n' + c if i % 6 == 0 else c, end=' | ')

nb_not_null = pd.DataFrame((~data.isna()).sum(axis=0), columns=['nb'])
nb_not_null.sort_values(by=['nb'], axis=0, ascending=True, inplace=True)
nb_not_null.T.head(150)

mask = pd.cut(nb_not_null['nb'], [-1, 0, 5, 10, 100, 300, 10000, 2000000])  #
mask.value_counts(normalize=False, sort=False)

data.drop(columns='countries', inplace=True)
data.drop(columns='labels', inplace=True)
data.drop(columns='traces', inplace=True)
data.drop(columns='additives', inplace=True)
data.drop(columns='categories', inplace=True)
data.drop(columns='main_category', inplace=True)

data.rename(columns={'nutriscore_score': 'nutriscore',
                     'nutriscore_grade': 'nutrigrade',
                     'traces_en': 'traces',
                     'labels_en': 'labels',
                     'pnns_groups_1': 'pnns1',
                     'pnns_groups_2': 'pnns2',
                     'nutrition_score_fr_100g': 'nutriscore_fr'},
            inplace=True)

data.rename(columns={'countries_en': 'countries',
                     'packaging_en': 'packaging',
                     'additives_en': 'additives',
                     'main_category_en': 'main_category',
                     'categories_en': 'categories',
                     'nutrition_score_fr_100g': 'nutriscore_fr'},
            inplace=True)

''' returns indices where both are filled, or first only, or second, or none'''


def comp_df(df, col1, col2, print_option):
    m_both = (~df[[col1, col2]].isna()).sum(axis=1) == 2
    m_one = (~df[[col1, col2]].isna()).sum(axis=1) == 1
    m_col1 = m_one & ~df[col1].isna()
    m_col2 = m_one & ~df[col2].isna()
    mnone = ~m_one & ~m_both
    ind_both = df.index[m_both].to_list()
    ind_col1 = df.index[m_col1].to_list()
    ind_col2 = df.index[m_col2].to_list()
    ind_none = df.index[mnone].to_list()
    if print_option:
        print("nb rows both filled: ", len(ind_both))
        print("nb rows with only", col1, "filled: ", len(ind_col1))
        print("nb rows with only", col2, "filled: ", len(ind_col2))
        print("nb rows not filled: ", len(ind_none))
    else:
        pass
    return (ind_both, ind_col1, ind_col2, ind_none)


li_col = ['packaging']
for c in li_col:
    c1 = c
    c2 = c + '_text'
    print('COLUMN', c, ':')
    t_ind = comp_df(data, c1, c2, True)
    data.loc[t_ind[1], [c1, c2]] = np.nan

data.drop(columns='packaging_text', inplace=True)

for i, c in enumerate(data.columns):
    print('\n' + c if i % 6 == 0 else c, end=' | ')

cols = ['brands', 'countries', 'labels', 'traces', 'additives',
        'allergens', 'main_category', 'categories', 'pnns2', 'pnns1']

thresh = [0, 2, 5, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 5000]

for c in cols:
    data[c] = data[c].replace([r'[-]'], [' '], regex=True)



cat_cols = ['main_category', 'pnns1', 'pnns2']


def filter_main_categories(ser, n):
    ser_m = ser
    flat_values = pd.Series([item for sublist in ser.str.split(',') \
                             for item in sublist])
    cat_occ = flat_values.value_counts()
    to_keep = cat_occ[cat_occ > n].index
    ser_m = ser_m.apply(lambda x: tuple([s.strip() for s in x.split(',') \
                                         if s.strip() in to_keep]))
    return ser_m


n = 15

print('Number of categories: ', [(cat + ' (' + str(data[cat].nunique()) + ') ') for cat in cols])

tot = data.shape[0]
ser = pd.DataFrame([[c, data[c].astype(str).str.contains('unknown').sum() * 100 / tot] \
                    for c in data.select_dtypes('object').columns],
                   columns=['col', 'pct_unknown']).set_index('col')
ser.sort_values('pct_unknown').plot.barh()

data.loc[data[data['product_name'].str.len() < 2].index, 'product_name'] = np.nan

data['quantity'] = data['quantity'].astype('object')


def sel_gr(li, li_prio1, li_prio2):
    res = 0
    nums, units = li
    tab_t_u = []
    ind = np.nan
    for i in units:
        if i in li_prio1:
            tab_t_u.append(2)
        elif i in li_prio2:
            tab_t_u.append(1)
        else:
            tab_t_u.append(0) if i != '' else tab_t_u.append(np.nan)
    i_tab1 = [i for i in range(len(tab_t_u)) if tab_t_u[i] == 2]  # indexes of all volumes (prio1)
    i_tab2 = [i for i in range(len(tab_t_u)) if tab_t_u[i] == 1]  # indexes of all masses (prio2)
    i_tab0 = [i for i in range(len(tab_t_u)) if tab_t_u[i] == 0]  # indexes of all others (prio3)
    if len(i_tab1) > 0:  # prio1 (vol)
        ind = i_tab1[np.argmax([nums[i] for i in i_tab1])]
    elif len(i_tab2) > 0:  # prio2 (mass)
        ind = i_tab2[np.argmax([nums[i] for i in i_tab2])]
    else:
        ind = i_tab0[np.argmax([nums[i] for i in i_tab0])] if len(i_tab0) > 0 else np.nan
    return (nums[ind], units[ind]) if ind is not np.nan else (np.nan, np.nan)


df_q = data['quantity'].dropna().to_frame()
print("Nb of notna values in 'quantity': {} on {}, i.e. {:.1f}%" \
      .format(df_q.shape[0], data.shape[0], df_q.shape[0] * 100 / data.shape[0]))

'''Cleaning 'quantity' column routines'''


def safe_exe(def_val, function, *args):
    try:
        return function(*args)
    except:
        return def_val


def conv_float(my_str):
    idx = 0
    if 'x' in my_str:
        idx = my_str.find('x')
        n1 = safe_exe(0, float, my_str[:idx])
        n2 = safe_exe(0, float, my_str[idx + 1:])
        return n1 * n2
    else:
        return safe_exe(0, float, my_str)


def num_units(my_str):
    my_str = my_str.lower().strip()
    regex = r'([0-9.,x ]*)\s*([^()0-9 !,\-±=\*\+/.-\?\[\]]*\s*)'
    res = re.findall(regex, my_str)
    res.remove(('', ''))
    num = [conv_float(gr[0].replace(' ', '').replace(',', '.')) for gr in res]
    unit = [gr[1].strip() for gr in res]
    res = list(zip(num, unit))
    return num, unit


li_u_mass = ['g', 'kg', 'gr', 'grammes', 'grs', 'st', 'mg', 'gramm', 'lb', 'gram',
             'grams', 'gramos', 'lbs', 'gm', 'lt', 'lts', 'gramme', 'kilo', '公克',
             'grammi', 'kgs', 'kgr', 'gms', 'g-', 'grms', 'pound', 'pounds',
             'ounces', 'ounce', 'grm', 'grames', 'غرام', 'جرام', 'غ', 'غم', 'جم',
             'g℮', 'г', 'кг', '克', 'грамм', 'גרם', 'kilogramm', 'gramas', 'γρ',
             'kilogrammae', 'livres', 'grame', 'kilos']
li_u_vol = ['ml', 'dl', 'l', 'cl', 'oz', 'litre', 'fl', 'litres', 'liter', 'litro',
            'litri', 'litr', 'ltr', 'lt', 'lts', 'gallon', 'half-gallon',
            'litros', 'litroe', 'liters', 'cc', 'kl', 'pint', 'pints', 'gal',
            'mls', 'centilitres', 'لتر', 'مل', 'ل', 'ليتر', 'มล', 'ลิตร', 'мл', 'л',
            'litrè', 'milliliter', 'millilitre', 'литр', 'литра', 'mml',
            'מ״ל', 'millilitres', 'λίτρο', 'mĺ', 'cm', 'cm³']

df_q['analysis'] = df_q['quantity'].apply(num_units)
df_q[['num_gr', 'unit_gr']] = pd.DataFrame(df_q['analysis'].tolist(),
                                           index=df_q['analysis'].index)

my_fun = lambda x: sel_gr(x, li_u_vol, li_u_mass)
df_q[['num', 'unit']] = pd.DataFrame(df_q['analysis'].apply(my_fun).tolist(),
                                     index=df_q['analysis'].index)
print(df_q.head(5).T)

d_mass_vol = dict([(u, 'mass') if u in li_u_mass else \
                       (u, 'vol') if u in li_u_vol else \
                           (u, 'other') for u in df_q['unit'].unique()])
df_q['unit_type'] = df_q['unit'].dropna().map(d_mass_vol)

df_q = df_q.dropna(subset=['unit', 'num'])

df_q = df_q[df_q['num'].between(0.0001, 10000)]

print("df before: ", data.shape,
      " | df_q: ", df_q.shape)
data = data.merge(df_q[['num', 'unit', 'unit_type']],
                  how='left', left_index=True, right_index=True)
data = data.rename(columns={'num': 'quantity_num',
                            'unit': 'quantity_unit',
                            'unit_type': 'quantity_type'})
data['quantity_type'].fillna('unknown')

print("df after: \n", data.shape)

del data['quantity']

for i, c in enumerate(data.columns):
    print('\n' + c, end=' | ') if (i) % 6 == 0 else print(c, end=' | ')

print(data.filter(like='quant')[(data.quantity_unit.notnull()) &
                                (data.quantity_unit != 'g') &
                                (data.quantity_unit != 'l')])

vol_1 = dict([(s, 1000) for s in ['litres', 'liter', 'litro', 'l', 'litre', 'λίτρο',
                                  'litrè', 'litri', 'litr', 'ltr', 'lt', 'lts',
                                  'litros', 'litroe', 'liters', 'لتر', 'ل', 'ليتر',
                                  'л', 'ลิตร', 'литр', 'литра']])
vol_2 = dict([(s, 1) for s in ['ml', 'mls', 'mls', 'مل', 'มล', 'мл',
                               'milliliter', 'millilitre', 'מ״ל', 'millilitres',
                               'mĺ', 'cm', 'cm³', 'cc']])
vol_3 = {'oz': 29.57, 'cl': 10, 'centilitres': 10, 'dl': 10,
         'gallon': 3.78541, 'gal': 3.78541, 'half-gallon': 1.89271}

dict_vol = dict(list(vol_1.items()) + list(vol_2.items()) + list(vol_3.items()))
dict_vol = dict([(k, (v, 'ml')) for k, v in dict_vol.items()])

mass_1 = dict([(s, 1000) for s in ['kg', 'kilo', 'кг', 'kilogrammae',
                                   'kilogramm', 'kilos', 'kgs', 'kgr']])
mass_2 = dict([(s, 1) for s in ['g', 'gr', 'grammes', 'grs', 'gramm', 'gram', 'grams',
                                'gramos', 'gm', 'gramme', '公克', 'γρ', 'grammi',
                                'gms', 'g-', 'grms', 'grm', 'grames', 'غرام', 'جرام',
                                'غ', 'غم', 'جم', 'g℮', 'г', '克', 'грамм', 'גרם',
                                'gramas', 'grame']])
mass_3 = dict([(s, 453.592) for s in ['lb', 'lbs', 'livres', 'pound', 'pounds']])
mass_4 = {'st': 6350.29, 'mg': 0.001, 'fl': 33.81, 'pint': 473.18,
          'pints': 473.18, 'ounces': 28.3495, 'ounce': 28.3495}
dict_mass = dict(list(mass_1.items()) + list(mass_2.items()) \
                 + list(mass_3.items()) + list(mass_4.items()))
dict_mass = dict([(k, (v, 'g')) for k, v in dict_mass.items()])

dict_m_v = dict(list(dict_vol.items()) + list(dict_mass.items()))

ser = data['quantity_unit'].map(dict_m_v).dropna(axis=0)
df_temp = pd.DataFrame(list(ser.values),
                       columns=['coef', 'quantity_unit_n'], index=ser.index)
df_temp['quantity_num_n'] = data['quantity_num'].mul(df_temp['coef'])

pd.concat([df_temp, data[['quantity_num', 'quantity_unit']]], axis=1).sample(5)

print("df before: ", data.shape)
data = data.merge(df_temp[['quantity_num_n', 'quantity_unit_n']],
                  how='left', left_index=True, right_index=True)
del data['quantity_num'], data['quantity_unit']
data = data.rename(columns={'quantity_num_n': 'quantity_num',
                            'quantity_unit_n': 'quantity_unit'})
print("df after: ", data.shape)

data.filter(like='quant')

print(data['quantity_num'].plot.box(vert=False))

data['quantity_num'] = data['quantity_num'].where(data['quantity_num'].between(0, 10001))

print(data['quantity_num'].fillna(0, inplace=True))

print(data['quantity_num'].plot.box(vert=False))

print((data['quantity_num'] == 0).sum())
print((data['quantity_num'].isna()).sum())

my_c = data.columns[data.columns.str.contains('serving')]
data[my_c][data[my_c].notna().any(axis=1)].sample(5)

del data['serving_size']

data['serving_quantity'] = data['serving_quantity'].clip(0, 600)
data['serving_quantity'].map({0: np.nan, 600: np.nan})
print(data['serving_quantity'].plot.box(vert=False))

data['quantity_unit'] = data['quantity_unit'].fillna('unknown')
data['quantity_unit'].unique()

data = add_feature_hashed_column(data, 'quantity_unit', n_features=3)

print('cat of pnns2/unknown in pnns1:\n',
      data[data['pnns1'] == 'unknown']['pnns2'].value_counts())
print('cat of pnns1/unknown in pnns2: ',
      data[data['pnns2'] == 'unknown']['pnns1'].value_counts())

data.pnns1.unique()
data.pnns2.unique()

data.groupby('pnns1').get_group('beverage')['pnns2'].unique()

i_bev = data.groupby('pnns1').get_group('beverage').index  # 57472
i_bev = i_bev.append(data.groupby('pnns2').get_group('alcoholic beverage').index)

data[data['pnns2'].isin(['sweetened beverage', 'fruit juice', 'unsweetened beverage', 'plant based milk substitute',
                         'tea and herbal tea and coffee', 'artificially sweetened beverage', 'water and flavored water',
                         'fruit nectar'])].shape, data[data['pnns1'] == 'beverage'].shape

pnns1 = ['fruit and vegetable', ]
pnns2 = ['fruit', 'vegetable', 'dried fruit', 'legume']

data.groupby('pnns1').get_group('fruit and vegetable')['pnns2'].unique()

print(data[data['pnns2'].isin(['fruit', 'vegetable', 'dried fruit', 'soup'])].shape, \
      data[data['pnns1'] == 'fruit and vegetable'].shape)

pnns1 = ['fat and sauce']
pnns2 = ['dressing and sauce', 'pizza pie and quiche', 'one dish meal', 'salty and fatty product',
         'fat', 'dairy dessert', 'breakfast cereal', 'appetizer', 'processed meat', 'sandwich']

data.groupby('pnns1').get_group('fat and sauce')['pnns2'].unique()




def hash_column(column, n_features=1000):
    iterable_column = [[str(val)] for val in column]

    hasher = FeatureHasher(n_features=n_features, input_type='string')
    hashed_features = hasher.transform(iterable_column)
    hashed_column = hashed_features.toarray()
    return hashed_column


def add_hashed_column(df, column_name, n_features=1000):
    hashed_column = hash_column(df[column_name], n_features=n_features)

    hashed_column_name = column_name + '_hashed'
    df[hashed_column_name] = pd.Series(hashed_column.tolist(), index=df.index)

    return df


import pandas as pd


def add_feature_hashed_column(df, column_name, n_features=1000):
    hashed_column = hash_column(df[column_name], n_features=n_features)

    # Add the hashed column to the DataFrame
    hashed_column_name = column_name + '_hashed'
    df[hashed_column_name] = pd.Series(hashed_column.tolist(), index=df.index)

    return df


hash_pnns1 = hash_column(data['pnns1'], n_features=11)
print(hash_pnns1.shape)
add_hashed_column(data, 'pnns1', n_features=11)
add_feature_hashed_column(data, 'pnns1', n_features=11)


def add_feature_hashed_column(df, column_name, n_features=1000):
    hasher = FeatureHasher(n_features=n_features, input_type='string')

    hashed_column = hasher.transform(df[column_name].apply(lambda x: [x]))

    hashed_df = pd.DataFrame(hashed_column.toarray())

    hashed_df.columns = [column_name + '_' + str(i) for i in range(n_features)]

    df = pd.concat([df, hashed_df], axis=1)

    return df


data = add_feature_hashed_column(data, 'pnns1', n_features=10)

data.filter(like='ingre')
data.rename(columns={'ingredients_text': 'ingredients'}, inplace=True)
IngCol = data['ingredients']

print('% of Null values = ', data['ingredients'].isnull().sum() / len(data) * 100, '%')

head_ing = data[~data['ingredients'].isnull()]['ingredients'].head(10)

for el in head_ing:
    print(el)
    print()

data.filter(like='categories')
print('% of Null values = ', data['categories'].isnull().sum() / len(data) * 100, '%')

head_ing = data[~data['categories'].isnull()]['categories'].head(10)

for el in head_ing:
    print(el)
    print()

nltk.download('popular')

data.select_dtypes(include=['object'])
CatValList = data['categories']
print(CatValList)

stop_words = set(stopwords.words('german'))
stop_words = set(stopwords.words('french'))
stop_words = set(stopwords.words('portuguese'))
stop_words = set(stopwords.words('spanish'))
stop_Words = set(stopwords.words('english'))

le = preprocessing.LabelEncoder()

data['nutrigrade'] = le.fit_transform(data['nutrigrade'])

data.select_dtypes(include=['object']).columns
data.drop(columns=['url', 'abbreviated_product_name',
                   'generic_name', 'packaging', 'brands', 'categories', 'origins_en',
                   'manufacturing_places', 'labels', 'emb_codes',
                   'first_packaging_code_geo', 'purchase_places', 'stores', 'countries',
                   'ingredients', 'allergens', 'traces', 'no_nutrition_data', 'additives',
                   'pnns1', 'pnns2', 'food_groups_en', 'brand_owner', 'ecoscore_grade',
                   'owner', 'main_category', 'quantity_type', 'quantity_unit',
                   'pnns1_hashed'], inplace=True)

for i, c in enumerate(data.columns):
    print('\n' + c if i % 6 == 0 else c, end=' | ')

print(data.shape)
data.to_csv('../data/df_CatVal_cleanedV2.csv')

df = data[['code', 'product_name', 'quantity_num']]
print(df)

df.to_csv('../data/dfCatVal_code_product_quantity.csv')

print(df.shape)
