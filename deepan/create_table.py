import pandas as pd

from loadclinical import load_clinical
from loadexpression import load_expression
from loadmutations import load_mutations


def create_binary_table(dataset, clinical, mutation, expression):  # entity = 'LUAD & LUSC', binary = True

    # loading all three datatypes and comparing patient IDS
    df_clin, df_y_all = load_clinical(dataset)
    df_mut = load_mutations(dataset)
    df_expr = load_expression(dataset)

    # count ids
    ids_clin = df_clin.index.values.tolist()  # 522
    ids_mut = df_mut.index.values.tolist()  # 567
    ids_expr = df_expr.index.values.tolist()  # 515

    '''
    # count features
    cols_clin = list(df_clin.columns)  # 22 features
    cols_mut = list(df_mut.columns)  # 18964 features
    cols_expr = list(df_mut.columns)  # 19572 features
    '''

    # get intersecting patient ids
    ids_clin_mut = [e for e in ids_mut if e in ids_clin]  # 515
    ids_clin_mut_expr = [e for e in ids_clin_mut if e in ids_expr]  # for LUAD 511 patients complete

    # subset for intersecting patients
    df1 = df_clin.loc[ids_clin_mut_expr].sort_index()
    df2 = df_mut.loc[ids_clin_mut_expr].sort_index()
    df3 = df_expr.loc[ids_clin_mut_expr].sort_index()

    print('Shapes of dfs: Clinical: ', df1.shape, ', Mutations: ', df2.shape, ', Expression: ', df3.shape)

    # choose selected data types
    dfs = []
    if clinical:
        dfs.append(df1)
    if mutation:
        dfs.append(df2)
    if expression:
        dfs.append(df3)
    df_all = pd.concat(dfs, axis=1)
    print('Concatenated dataset binary:', df_all.shape)

    #subset df_y_all and order for grouping later
    df_y = df_y_all.loc[ids_clin_mut_expr].sort_index()

    # save file
    #df_all.to_csv(r'/media/administrator/INTERNAL3_6TB/TCGA_data/all_binary_selected.txt', index=True, sep='\t')

    return df_all, df_y