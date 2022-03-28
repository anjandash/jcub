#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import mpl_toolkits

train_data = pd.read_csv("./datasets/base/jcub_train_full_scale_csv.csv", header=0)
test_data = pd.read_csv("./datasets/base/jcub_test_full_scale_csv.csv", header=0)

data = pd.concat([train_data, test_data], ignore_index=True)
data.drop(columns=["method_id", "method_text"], inplace=True)

# #%%
# data.describe()
# # %%

# data_0 = data[data["cmpx_label"] == 0]
# data_0.describe()
# # %%
# data['nuid_label'].value_counts().plot(kind='bar')
# plt.title('Number of Unique Identifiers')
# plt.xlabel('Number of Unique Identifiers')
# plt.ylabel('Count')
# sns.despine
# # %%

# plt.scatter(data.cmpx_label, data.nmtk_label)
# plt.title("CMPX vs Number of Tokens")
# # %%
# plt.scatter(data.cmpx_label, data.nuid_label)
# plt.title("CMPX vs Number of Unique Identifiers")

# #%%
# plt.scatter(data.cmpx_label, data.ntid_label)
# plt.title("CMPX vs Number of Total Identifiers")

# #%%
# plt.scatter(data.cmpx_label, data.nmlt_label)
# plt.title("CMPX vs Number of Literals")

# #%%
# plt.scatter(data.cmpx_label, data.mxin_label)
# plt.title("CMPX vs Max Indent")

# #%%
# plt.scatter(data.cmpx_label, data.sloc_label)
# plt.title("CMPX vs Source Lines of Code")

# #%%
# plt.scatter(data.cmpx_label, data.tloc_label)
# plt.title("CMPX vs Total Lines of Code")

# #%%
# plt.scatter(data.cmpx_label, data.nmop_label)
# plt.title("CMPX vs Number of Operators")

# #%%
# plt.scatter(data.cmpx_label, data.nmrt_label)
# plt.title("CMPX vs Number of Returns")

# #%%
# plt.scatter(data.cmpx_label, data.nmpr_label)
# plt.title("CMPX vs Number of Parameters")
# %%
# print(data.columns)
# %%
# Labels
# ------
# 'nuid_label', xxxx
# 'mxin_label', xxxx
# 'nmlt_label', xxxx
# 'cmpx_label', ----
# 'sloc_label', xxxx
# 'nmpr_label', xxxx
# 'nmtk_label', xxxx
# 'nmrt_label', xxxx
# 'tloc_label', xxxx
# 'nmop_label', xxxx
# 'ntid_label', xxxx
# 'name_label'  ----

#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
reg = LinearRegression() #ls=least squares

labels = data['cmpx_label']
train1 = data.drop(['name_label', 'cmpx_label'], axis=1)

x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size=0.20, shuffle=False) #random_state=2) 

reg.fit(x_train, y_train)
#reg.score(x_test, y_test)
pred_as = reg.predict(x_test)

#%%
from sklearn.metrics import r2_score
print(r2_score(y_test, pred_as))


# %%
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls') #ls=least squares #squared_error

clf.fit(x_train, y_train)
#clf.score(x_test,y_test)
pred_as = clf.predict(x_test)


#%%
from sklearn.metrics import r2_score
print(r2_score(y_test, pred_as))

# %%
