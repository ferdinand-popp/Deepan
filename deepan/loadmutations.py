import pandas as pd


def preselect_mutations(df):
    # choose features on most present -> sum col added
    df['Sum'] = df.sum(axis=1)

    # sort by sum
    df = df.sort_values(by='Sum', ascending=False)

    # subsetting for top 100 ones & drop sum
    df_selected = df.head(100)
    df_preselected = df_selected.drop(['Sum'], axis=1)
    return df_preselected


def load_mutations():
    url = r'/media/administrator/INTERNAL3_6TB/TCGA_data/somatic_mutation/LUAD/LUAD_somatic_mutation_snp_indel_mutect2_binary.txt'

    df_raw = pd.read_csv(url, sep='\t', index_col=0)
    df = preselect_mutations(df_raw)
    df_T = df.T

    # inspect
    cols_ids = list(df_T.columns)  # patients
    genes = list(df_T.index)  #

    return df_T
