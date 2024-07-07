import pandas as pd

pd.options.display.max_seq_items = 3000000
# Viz
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "/media/kirrog/data/data/infr_bd/en.openfoodfacts.org.products.csv"
nrows = 12000000
data = pd.read_csv(file_path, sep="\t", encoding="utf-8", low_memory=False)
print(data.head(3))

for i, c in enumerate(data.columns):
    print('\n' + c if i % 6 == 0 else c, end=' | ')


def print_null_pct(df):
    tot_null = df.isna().sum().sum()
    print('nb of null: ', tot_null, '\npct of null: ',
          '{:.1f}'.format(tot_null * 100 / (df.shape[0] * df.shape[1])))


print_null_pct(data)


def null_factor(data, tx_threshold=50):
    null_rate = ((data.isnull().sum() / data.shape[0]) * 100).sort_values(ascending=False).reset_index()
    null_rate.columns = ['Variable', "Null_Value"]
    high_null_rate = null_rate[null_rate.Null_Value >= tx_threshold]
    return high_null_rate


filling_features = null_factor(data, 0)
filling_features["Null_Value"] = 100 - filling_features["Null_Value"]
filling_features = filling_features.sort_values("Null_Value", ascending=False)

sup_threshold = 25

fig = plt.figure(figsize=(20, 35))

font_title = {'family': 'serif',
              'color': '#124b98',
              'weight': 'bold',
              'size': 13,
              }

sns.barplot(x="Null_Value", y="Variable", data=filling_features, palette="tab10")

plt.axvline(x=sup_threshold, linewidth=2, color='r')
plt.text(sup_threshold + 2, 65, 'Number of nulls to choose', fontsize=16, color='r')

plt.title("Number of nulls for each category (%)", fontdict=font_title)
plt.xlabel("Number of nulls (%)")
plt.show()