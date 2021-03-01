import pandas as pd


def preselect_mutations(df):
    # choose features on most present -> sum col added
    df['Sum'] = df.sum(axis=1)

    # sort by sum
    df = df.sort_values(by='Sum', ascending=False)

    # subsetting for top 100 ones & drop sum
    df_selected = df.head(200)
    df_preselected = df_selected.drop(['Sum'], axis=1)
    return df_preselected


def load_mutations():
    url1 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/somatic_mutation/LUAD/LUAD_somatic_mutation_snp_indel_mutect2_binary.txt'
    url2 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/somatic_mutation/LUSC/LUSC_somatic_mutation_snp_indel_mutect2_binary.txt'
    df_raw1 = pd.read_csv(url1, sep='\t', index_col=0)
    df_raw2 = pd.read_csv(url2, sep='\t', index_col=0)
    df_raw = df_raw1.append(df_raw2)
    df = preselect_mutations(df_raw)
    df_T = df.T

    # inspect
    cols_ids = list(df_T.columns)  # patients
    genes = list(df_T.index)  #
    df_T = df_T.dropna(axis=0, how='any')
    return df_T
