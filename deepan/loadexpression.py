import pandas as pd


def get_de():
    # uses the expression file to get most differentially expressed genes and returns list with gene names
    url = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_gene_expression_tpm.txt'
    df = pd.read_csv(url, sep='\t', index_col=0)

    ser = df.var(axis=1)

    #highest standarddeviation

    # sort by var
    ser = ser.sort_values(ascending=False)

    # subsetting for top 100 ones & drop var
    ser_selected = ser.head(200)

    # get indices
    feature_list = list(ser_selected.index.values)
    return feature_list


def preselect_expression(df):
    # input feature list
    features = get_de()
    df_preselected = df.loc[features].sort_index()

    return df_preselected


def load_expression():
    url_expression1 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_avr_gene_expression_binary.txt'
    url_expression2 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUSC/LUSC_avr_gene_expression_binary.txt'

    df_raw1 = pd.read_csv(url_expression1, sep='\t', index_col=0)
    df_raw2 = pd.read_csv(url_expression2, sep='\t', index_col=0)

    df_raw = df_raw1.append(df_raw2)

    df = preselect_expression(df_raw)
    df_T = df.T

    # inspect
    cols_ids = list(df_T.columns)  # 515 patients
    genes = list(df_T.index)  # 19572

    return df_T
