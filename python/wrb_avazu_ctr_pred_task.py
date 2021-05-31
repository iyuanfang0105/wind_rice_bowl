# %%
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

# %%
dataset_dir = '/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/'
train = pd.read_csv(os.path.join(dataset_dir,'data_24.csv'),compression='gzip')
test = pd.read_csv(os.path.join(dataset_dir, 'data_25.csv'), compression='gzip')
print('train shape: {}, test shape: {}'.format(train.shape, test.shape))


# def data_exploring():
#     click_distribution = train['click'].value_counts()
#     print('un_click_ratio: {}, click_ratio: {}'.format(click_distribution[0] / sum(click_distribution),
#                                                        click_distribution[1] / sum(click_distribution)))
#
#     # %%
#     # hour of day
#     train['hour_of_day'] = train['hour'].apply(lambda x: int(str(x)[-2:]))
#     train.groupby('hour_of_day').agg({'click':'sum'}).plot(figsize=(12,6))
#     plt.ylabel('Number of clicks')
#     plt.title('click trends by hour of day')
#     plt.show()
#
#     # %%
#     train.groupby(['hour_of_day', 'click']).size().unstack().plot(kind='bar', title="Hour of Day", figsize=(12,6))
#     plt.ylabel('count')
#     plt.title('Hourly impressions vs. clicks');
#     plt.show()
#
#     # %%
#     # hourly ctr
#     df_click = train[train['click'] == 1]
#     df_hour = train[['hour_of_day','click']].groupby(['hour_of_day']).count().reset_index()
#     df_hour = df_hour.rename(columns={'click': 'impressions'})
#     df_hour['clicks'] = df_click[['hour_of_day','click']].groupby(['hour_of_day']).count().reset_index()['click']
#     df_hour['CTR'] = df_hour['clicks']/df_hour['impressions']*100
#
#     plt.figure(figsize=(12,6))
#     sns.barplot(y='CTR', x='hour_of_day', data=df_hour)
#     plt.title('Hourly CTR');
#     plt.show()
#
#     # %%
#     # C1 value = 1005 has the most data, almost 92%.
#     # And then we can calculate the CTR of each C1 value.
#     print(train.C1.value_counts()/len(train))
#     C1_values = train.C1.unique()
#     C1_values.sort()
#     ctr_avg_list=[]
#     for i in C1_values:
#         ctr_avg=train.loc[np.where((train.C1 == i))].click.mean()
#         ctr_avg_list.append(ctr_avg)
#         print("{}: click through rate: {}".format(i,ctr_avg))
#
#     # %%
#     train.groupby(['C1', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='C1 histogram');
#     plt.show()
#
#     # %%
#     df_c1 = train[['C1','click']].groupby(['C1']).count().reset_index()
#     df_c1 = df_c1.rename(columns={'click': 'impressions'})
#     df_c1['clicks'] = df_click[['C1','click']].groupby(['C1']).count().reset_index()['click']
#     df_c1['CTR'] = df_c1['clicks']/df_c1['impressions']*100
#
#     plt.figure(figsize=(12,6))
#     sns.barplot(y='CTR', x='C1', data=df_c1)
#     plt.title('CTR by C1');
#     plt.show()
#
#     # %%
#     train['click'].mean()
#     df_c1.CTR.describe()
#
#     # %%
#     print(train.banner_pos.value_counts()/len(train))
#
#     banner_pos = train.banner_pos.unique()
#     banner_pos.sort()
#     ctr_avg_list=[]
#     for i in banner_pos:
#         ctr_avg=train.loc[np.where((train.banner_pos == i))].click.mean()
#         ctr_avg_list.append(ctr_avg)
#         print("{}: click through rate: {}".format(i,ctr_avg))
#
#     train.groupby(['banner_pos', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='banner position histogram');
#     plt.show()
#
#     # %%
#     df_banner = train[['banner_pos','click']].groupby(['banner_pos']).count().reset_index()
#     df_banner = df_banner.rename(columns={'click': 'impressions'})
#     df_banner['clicks'] = df_click[['banner_pos','click']].groupby(['banner_pos']).count().reset_index()['click']
#     df_banner['CTR'] = df_banner['clicks']/df_banner['impressions']*100
#     sort_banners = df_banner.sort_values(by='CTR',ascending=False)['banner_pos'].tolist()
#     plt.figure(figsize=(12,6))
#     sns.barplot(y='CTR', x='banner_pos', data=df_banner, order=sort_banners)
#     plt.title('CTR by banner position');
#     plt.show()
#
#     # %%
#     df_banner.CTR.describe()
#
#     # %%
#     print("There are {} sites in the data set".format(train.site_id.nunique()))
#     print('The top 10 site ids that have the most impressions')
#     print((train.site_id.value_counts()/len(train))[0:10])
#
#     top10_ids = (train.site_id.value_counts()/len(train))[0:10].index
#     click_avg_list=[]
#
#     for i in top10_ids:
#         click_avg=train.loc[np.where((train.site_id == i))].click.mean()
#         click_avg_list.append(click_avg)
#         print("for site id value: {},  click through rate: {}".format(i,click_avg))
#
#     # %%
#     top10_sites = train[(train.site_id.isin((train.site_id.value_counts()/len(train))[0:10].index))]
#     top10_sites_click = top10_sites[top10_sites['click'] == 1]
#     top10_sites.groupby(['site_id', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 site ids histogram');
#     plt.show()
#
#     # %%
#     df_site = top10_sites[['site_id','click']].groupby(['site_id']).count().reset_index()
#     df_site = df_site.rename(columns={'click': 'impressions'})
#     df_site['clicks'] = top10_sites_click[['site_id','click']].groupby(['site_id']).count().reset_index()['click']
#     df_site['CTR'] = df_site['clicks']/df_site['impressions']*100
#     sort_site = df_site.sort_values(by='CTR',ascending=False)['site_id'].tolist()
#     plt.figure(figsize=(12,6))
#     sns.barplot(y='CTR', x='site_id', data=df_site, order=sort_site)
#     plt.title('CTR by top 10 site id');
#     plt.show()
#
#     # %%
#     print("There are {} site domains in the data set".format(train.site_domain.nunique()))
#     print('The top 10 site domains that have the most impressions')
#     print((train.site_domain.value_counts()/len(train))[0:10])
#
#     top10_domains = (train.site_domain.value_counts()/len(train))[0:10].index
#     click_avg_list=[]
#
#     for i in top10_domains:
#         click_avg=train.loc[np.where((train.site_domain == i))].click.mean()
#         click_avg_list.append(click_avg)
#         print("for site domain value: {},  click through rate: {}".format(i,click_avg))
#
#     # %%
#     top10_domain = train[(train.site_domain.isin((train.site_domain.value_counts()/len(train))[0:10].index))]
#     top10_domain_click = top10_domain[top10_domain['click'] == 1]
#     top10_domain.groupby(['site_domain', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 site domains histogram');
#     plt.show()
#
#     # %%
#     df_domain = top10_domain[['site_domain','click']].groupby(['site_domain']).count().reset_index()
#     df_domain = df_domain.rename(columns={'click': 'impressions'})
#     df_domain['clicks'] = top10_domain_click[['site_domain','click']].groupby(['site_domain']).count().reset_index()['click']
#     df_domain['CTR'] = df_domain['clicks']/df_domain['impressions']*100
#     sort_domain = df_domain.sort_values(by='CTR',ascending=False)['site_domain'].tolist()
#     plt.figure(figsize=(12,6))
#     sns.barplot(y='CTR', x='site_domain', data=df_domain, order=sort_domain)
#     plt.title('CTR by top 10 site domain');
#     plt.show()
#
#     # %%
#     print("There are {} site categories in the data set".format(train.site_category.nunique()))
#     print('The top 10 site categories that have the most impressions')
#     print((train.site_category.value_counts()/len(train))[0:10])
#
#
#     top10_categories = (train.site_category.value_counts()/len(train))[0:10].index
#     click_avg_list=[]
#
#     for i in top10_categories:
#         click_avg=train.loc[np.where((train.site_category == i))].click.mean()
#         click_avg_list.append(click_avg)
#         print("for site category value: {},  click through rate: {}".format(i,click_avg))
#
#     top10_category = train[(train.site_category.isin((train.site_category.value_counts()/len(train))[0:10].index))]
#     top10_category_click = top10_category[top10_category['click'] == 1]
#     top10_category.groupby(['site_category', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 site categories histogram');
#     plt.show()
#
#     df_category = top10_category[['site_category','click']].groupby(['site_category']).count().reset_index()
#     df_category = df_category.rename(columns={'click': 'impressions'})
#     df_category['clicks'] = top10_category_click[['site_category','click']].groupby(['site_category']).count().reset_index()['click']
#     df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
#     sort_category = df_category.sort_values(by='CTR',ascending=False)['site_category'].tolist()
#     plt.figure(figsize=(12,6))
#     sns.barplot(y='CTR', x='site_category', data=df_category, order=sort_category)
#     plt.title('CTR by top 10 site category');
#     plt.show()
#
#     # %%
#     print("There are {} devices in the data set".format(train.device_id.nunique()))
#     print('The top 10 devices that have the most impressions')
#     print((train.device_id.value_counts()/len(train))[0:10])
#
#     top10_devices = (train.device_id.value_counts()/len(train))[0:10].index
#     click_avg_list=[]
#
#     for i in top10_devices:
#         click_avg=train.loc[np.where((train.device_id == i))].click.mean()
#         click_avg_list.append(click_avg)
#         print("for device id value: {},  click through rate: {}".format(i,click_avg))
#
#     top10_device = train[(train.device_id.isin((train.device_id.value_counts()/len(train))[0:10].index))]
#     top10_device_click = top10_device[top10_device['click'] == 1]
#     top10_device.groupby(['device_id', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 device ids histogram');
#     plt.show()
#
#     # %%
#     print("There are {} device ips in the data set".format(train.device_ip.nunique()))
#     print("There are {} device types in the data set".format(train.device_type.nunique()))
#     print("There are {} device models in the data set".format(train.device_model.nunique()))
#     print("There are {} device cnn types in the data set".format(train.device_conn_type.nunique()))
#
#     # %%
#     print('The impressions by device types')
#     print((train.device_type.value_counts()/len(train)))
#
#     train[['device_type','click']].groupby(['device_type','click']).size().unstack().plot(kind='bar', title='device types');
#     plt.show()
#
#     df_click[df_click['device_type']==1].groupby(['hour_of_day', 'click']).size().unstack().plot(kind='bar', title="Clicks from device type 1 by hour of day", figsize=(12,6));
#     plt.show()

# %%
train['flag'] = 'train'
test['flag'] = 'test'
dataset = pd.concat([train, test], axis=0).reset_index(drop=True)
print('dataset shape: {}'.format(dataset.shape))

# %%
# get the dataset features information, and make it following libffm format
columns_name = dataset.columns.values.tolist()
print('columns_name: {}'.format(columns_name))

primary_keys = ['hour', 'C1', 'banner_pos', 'site_id', 'site_category', 'app_domain', 'app_category',
                'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

secondary_keys = []
for col in primary_keys:
    col_v_unique = dataset[col].unique()
    col_v_unique = [col + '##' + str(i) for i in col_v_unique]
    secondary_keys.extend(col_v_unique)
    print('feature: ##{}##, values_count: {}, values: {}'.format(col, len(col_v_unique), col_v_unique))
print('len_primary_keys: {}, len_secondary_keys: {}'.format(len(primary_keys), len(secondary_keys)))

primary_keys_dict = {k: v for v, k in enumerate(primary_keys)}
secondary_keys_dict = {k: v for v, k in enumerate(secondary_keys)}

# %%
# reconstruct the dataset
columns_name_selected = primary_keys + ['flag', 'click']
print('columns_name_selected: {}'.format(columns_name_selected))
dataset_selected = dataset[columns_name_selected]
print('dataset_selected shape: {}'.format(dataset_selected.shape))

# %%
# feature enginnering as one-hot
# feature enginnering as ffm

# for pk in primary_keys_dict.keys():
#     dataset_selected[pk+'_ffm'] = dataset_selected[pk].apply(lambda x: str(primary_keys_dict[pk]) + ':'
#                                          + str(secondary_keys_dict[pk + '##' + str(x)]) + ':' + str(1))
#     break
# print(dataset_selected.head())

ffm_train_dataset_save_path = '/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/ffm_train.txt'
ffm_test_dataset_save_path = '/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/ffm_test.txt'
test_labels = []
with open(ffm_train_dataset_save_path, mode='w', encoding='utf-8') as ftrain:
    with open(ffm_test_dataset_save_path, mode='w', encoding='utf-8') as ftest:
        for index, row in dataset_selected.iterrows():
            row_dict = row.to_dict()

            row_str = ''
            label = row_dict['click']
            row_str += str(label) + ' '
            for k in row_dict.keys():
                if k != 'click' and k != 'flag':
                    sk = k + '##' + str(row_dict[k])
                    # print(primary_keys_dict[k], secondary_keys_dict[sk])
                    row_str += str(primary_keys_dict[k])+':'+str(secondary_keys_dict[sk]) + ':1' + ' '
                # break

            ffm_d_str = row_str.strip() + '\n'

            if row_dict['flag'] == 'train':
                ftrain.write(ffm_d_str)
            if row_dict['flag'] == 'test':
                ftest.write(ffm_d_str)

            if index % 10000 == 0:
                print('convert data to ffm format: {} / {}'.format(index, dataset_selected.shape[0]))

# %%
# build model
import xlearn as xl
fm_model = xl.create_ffm()
fm_model.setTrain('/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/ffm_train.txt')
fm_model.setValidate('/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/ffm_test.txt')
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
fm_model.fit(param, "/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/python/model.out")

# %%
# fm_model.cv(param)

# %%
fm_model.setSigmoid()
fm_model.setTest("/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/ffm_test.txt")
fm_model.predict("/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/python/model.out", "./pred_probs.txt")

# fm_model.setSign()
# fm_model.setTest("/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/ffm_test.txt")
# fm_model.predict("/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/python/model.out", "./preds.txt")

# %%
# preds = []
# with open('./preds.txt', 'rb') as fin:
#     for l in fin.readlines():
#         preds.append(float(l.strip()))

preds = []
pred_probs = []
with open('./pred_probs.txt', 'rb') as fin:
    for l in fin.readlines():
        res = float(l.strip())
        if res >= 0.5:
            preds.append(1.0)
        else:
            preds.append(0)
        pred_probs.append(res)
print(preds, pred_probs)

# %%
labels = []
with open('/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/ffm_test.txt', 'rb') as fin:
    for l in fin.readlines():
        labels.append(float(l.strip().split()[0]))
        # break
print(labels)

# %%
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(labels, preds))
print(roc_auc_score(labels, pred_probs))

# %%
# test on criteo_ctr small dataset
fm_model = xl.create_ffm()
fm_model.setTrain('/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/data/criteo_ctr/small_train.txt')
fm_model.setValidate('/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/data/criteo_ctr/small_test.txt')
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
fm_model.fit(param, "/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/python/model.out")

# %%
# fm_model.cv(param)

# %%
fm_model.setSigmoid()
fm_model.setTest("/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/data/criteo_ctr/small_test.txt")
fm_model.predict("/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/python/model.out", "./pred_probs.txt")

# fm_model.setSign()
# fm_model.setTest("/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl//data/avazu_ctr/train/ffm_test.txt")
# fm_model.predict("/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/python/model.out", "./preds.txt")

# %%
# preds = []
# with open('./preds.txt', 'rb') as fin:
#     for l in fin.readlines():
#         preds.append(float(l.strip()))

preds = []
pred_probs = []
with open('./pred_probs.txt', 'rb') as fin:
    for l in fin.readlines():
        res = float(l.strip())
        if res >= 0.5:
            preds.append(1.0)
        else:
            preds.append(0)
        pred_probs.append(res)
print(preds, pred_probs)

# %%
labels = []
with open('/Users/wind/WORK/wind_rice_bowl/code/wind_rice_bowl/data/criteo_ctr/small_test.txt', 'rb') as fin:
    for l in fin.readlines():
        labels.append(float(l.strip().split()[0]))
        # break
print(labels)

# %%
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(labels, preds))
print(roc_auc_score(labels, pred_probs))