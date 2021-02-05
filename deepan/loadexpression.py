import pandas as pd


def get_de():
    # uses the expression file to get most differentially expressed genes and returns list with gene names
    url = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_gene_expression_tpm.txt'
    df = pd.read_csv(url, sep='\t', index_col=0)

    ser = df.var(axis=1)

    # sort by var
    ser = ser.sort_values(ascending=False)

    # subsetting for top 1000 ones & drop var
    ser_selected = ser.head(1000)

    # get indices
    feature_list = list(ser_selected.index.values)
    return feature_list


def preselect_expression(df):
    # input feature list
    features = get_de()
    df_preselected = df.loc[features].sort_index()
    print(df_preselected)

    return df_preselected


def load_expression():
    url_expression = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_avr_gene_expression_binary.txt'

    df_raw = pd.read_csv(url_expression, sep='\t', index_col=0)

    df = preselect_expression(df_raw)
    df_T = df.T

    # inspect
    cols_ids = list(df_T.columns)  # 515 patients
    genes = list(df_T.index)  # 19572

    return df_T
