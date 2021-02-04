import pandas as pd
def load_expression():
    url_expression = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_avr_gene_expression_binary.txt'

    df_raw = pd.read_csv(url_expression, sep='\t', index_col=0)



    # inspect
    cols_ids = list(df_raw.columns)  # 515 patients
    genes = list(df_raw.index)  #19572

    df = df_raw.T
    return df



