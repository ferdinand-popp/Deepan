import pandas as pd
def load_mutations():
    url = r'/media/administrator/INTERNAL3_6TB/TCGA_data/somatic_mutation/LUAD/LUAD_somatic_mutation_snp_indel_mutect2_binary.txt'

    df_raw = pd.read_csv(url, sep='\t', index_col=0)

    # inspect
    cols_ids = list(df_raw.columns)  # patients
    genes = list(df_raw.index)  #

    df = df_raw.T
    #print(df.head())
    return df



