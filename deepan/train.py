# test script for LUAD
import pandas as pd
import numpy as np
from loadclinical import load_clinical
from loadmutations import load_mutations
from loadexpression import load_expression

# loading all three datatypes and comparing patient IDS

df_clin = load_clinical()
df_mut = load_mutations()
df_expr = load_expression()

ids_clin = df_clin.index.values.tolist()  # 522
ids_mut = df_mut.index.values.tolist()  # 567
ids_expr = df_expr.index.values.tolist()  # 515

cols_clin = list(df_clin.columns)  # 22 features
cols_mut = list(df_mut.columns)  # 18964 features
cols_expr = list(df_mut.columns)  # 19572 features

ids_clin_mut = [e for e in ids_mut if e in ids_clin]  # 515
ids_clin_mut_expr = [e for e in ids_clin_mut if e in ids_expr]  # 511

# for LUAD 511 patients complete

# combine into large df
df1 = df_clin.loc[ids_clin_mut_expr].sort_index()
df2 = df_mut.loc[ids_clin_mut_expr].sort_index()
df3 = df_expr.loc[ids_clin_mut_expr].sort_index()

print(df1.shape, df2.shape, df3.shape)

df_all = pd.concat([df1, df2, df3], axis=1)
print(df_all.shape)

print('help')
# save file
df_all.to_csv(r'/media/administrator/INTERNAL3_6TB/TCGA_data/all_binary_selected.txt', index = True, sep='\t')
