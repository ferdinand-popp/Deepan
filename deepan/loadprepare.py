import pandas as pd

#url_clinical = r'media/administrator/INTERNAL3_6TB/TCGA_data/clinical_data/LUSC/LUSC_clinical_data_firebrowser_20210125.txt'
url_clinical = r'media/administrator/INTERNAL3_6TB/TCGA_data/clinical_data/LUAD/LUAD_clinical_data_firebrowser_20210125.txt'

df_raw = pd.read_csv(url_clinical, sep='\t',
                     index_col=0)  # otherwise pd.read_fwf('file.txt', sep='\s{2,}', header=[0],skiprows=[1])

# inspect
cols = list(df_raw.columns)  # LUSC 20 #LUAD 20
IDs = list(df_raw.patient)  # LUSC 504 #LUAD 522

print(df_raw.shape)
print(cols, IDs)