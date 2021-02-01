import pandas as pd

url_expression = r'/media/administrator/INTERNAL3_6TB/TCGA_data/gene_expression/LUAD/LUAD_avr_gene_expression_binary.txt'

df_raw = pd.read_csv(url_expression, sep='\t', index_col=0)  # otherwise pd.read_fwf('file.txt', sep='\s{2,}', header=[0],skiprows=[1])


print(df_raw.shape) #

# inspect
cols_ids = list(df_raw.columns)  # 515 patients
genes = list(df_raw.index)  #19572

#check types of cols and adapt them
#print(df_raw.info(verbose = True))

#look at statistics
#nuller = df_raw.isnull().sum() / complete
#print(nuller)



