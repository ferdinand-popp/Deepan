import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preselect_expression(df_all, dataset):
    if dataset == 'LUAD':
        # uses the expression file to get most differentially expressed genes and returns list with gene names
        url = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_gene_expression_tpm.txt'
        df = pd.read_csv(url, sep='\t', index_col=0)  # 20527 rows and 515 patient cols
    if dataset == 'LUSC':
        # uses the expression file to get most differentially expressed genes and returns list with gene names
        url = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUSC/LUSC_gene_expression_tpm.txt'
        df = pd.read_csv(url, sep='\t', index_col=0)  # 20527 rows and 515 patient cols
    if dataset == 'NSCLC':
        # uses the expression file to get most differentially expressed genes and returns list with gene names
        url1 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_gene_expression_tpm.txt'
        url2 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUSC/LUSC_gene_expression_tpm.txt'

        df1 = pd.read_csv(url1, sep='\t', index_col=0)
        df2 = pd.read_csv(url2, sep='\t', index_col=0)

        df = pd.concat([df1, df2], axis=1, join='inner')

    ser = df.var(axis=1)

    #PCA
    PCA = False
    if PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=100)
        principalComponents = pca.fit_transform(df.T)
        min_max_scaler = MinMaxScaler()  # or robust
        np_scaled = min_max_scaler.fit_transform(principalComponents)  # to scale Gene wise and not patient wise

        principalDf = pd.DataFrame(data=np_scaled, index=df.columns)

        df_final = principalDf.T
        return df_final
    # highest standarddeviation

    # sort by var
    ser = ser.sort_values(ascending=False)

    # subsetting for top ones & drop var
    ser_selected = ser.head(100)

    # get indices
    features = list(ser_selected.index.values)

    # subset for the wanted patients
    numerical = True
    if numerical:
        df_preselected = df.loc[features].sort_index()

        #Scale
        min_max_scaler = MinMaxScaler() #or robust
        np_scaled = min_max_scaler.fit_transform(df_preselected.T)  # to scale Gene wise and not patient wise
        df_normalized = pd.DataFrame(np_scaled, columns=list(df_preselected.T.columns), index=list(df_preselected.columns))
        df_final = df_normalized.T
    else:
        df_final = df_all.loc[features].sort_index()

    return df_final


def load_expression(dataset):
    if dataset == 'LUAD':
        url_expression = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_avr_gene_expression_binary.txt'
        df_raw = pd.read_csv(url_expression, sep='\t', index_col=0)
    if dataset == 'LUSC':
        url_expression = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUSC/LUSC_avr_gene_expression_binary.txt'
        df_raw = pd.read_csv(url_expression, sep='\t', index_col=0)
    if dataset == 'NSCLC':
        url_expression1 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_avr_gene_expression_binary.txt'
        url_expression2 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUSC/LUSC_avr_gene_expression_binary.txt'

        df_raw1 = pd.read_csv(url_expression1, sep='\t', index_col=0)
        df_raw2 = pd.read_csv(url_expression2, sep='\t', index_col=0)

        df_raw = pd.concat([df_raw1, df_raw2], axis=1, join='inner')

    # can be done by frequency or '/media/administrator/INTERNAL3_6TB/TCGA_data/gene_signatures'
    df = preselect_expression(df_raw, dataset)
    df_T = df.T

    # inspect
    # cols_ids = list(df_T.columns)  # 515 patients
    # genes = list(df_T.index)  # 19572

    # drop rows with missing values
    df_T = df_T.dropna(axis=0, how='any')

    return df_T
