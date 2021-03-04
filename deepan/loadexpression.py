import pandas as pd


def preselect_expression(df, dataset):
    if dataset == 'LUAD':
        # uses the expression file to get most differentially expressed genes and returns list with gene names
        url = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_gene_expression_tpm.txt'
        df = pd.read_csv(url, sep='\t', index_col=0)  # 20527 rows and 515 patient cols

    if dataset == 'NSCLC':
        # uses the expression file to get most differentially expressed genes and returns list with gene names
        url1 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_gene_expression_tpm.txt'
        url2 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUSC/LUSC_gene_expression_tpm.txt'

        df1 = pd.read_csv(url1, sep='\t', index_col=0)
        df2 = pd.read_csv(url2, sep='\t', index_col=0)

        df = pd.concat([df1, df2], axis=1, join='inner')

    ser = df.var(axis=1)

    # highest standarddeviation

    # sort by var
    ser = ser.sort_values(ascending=False)

    # subsetting for top ones & drop var
    ser_selected = ser.head(100)

    # get indices
    features = list(ser_selected.index.values)

    # subset for the wanted patients
    df_preselected = df.loc[features].sort_index()

    return df_preselected


def load_expression(dataset):
    if dataset == 'LUAD':
        url_expression = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_avr_gene_expression_binary.txt'
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
