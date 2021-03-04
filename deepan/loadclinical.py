import pandas as pd
import numpy as np


def load_clinical(dataset, url_clinical=None):
    if dataset == 'LUAD':
        if url_clinical is None:
            url_clinical = r'/media/administrator/INTERNAL3_6TB/TCGA_data/clinical_data/LUAD/LUAD_clinical_data_firebrowse_20210125.txt'
        df_raw = pd.read_csv(url_clinical, sep='\t', index_col=0)
    if dataset == 'NSCLC':
        if url_clinical is None:
            url_clinical1 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/clinical_data/LUAD/LUAD_clinical_data_firebrowse_20210125.txt'
            url_clinical2 = r'/media/administrator/INTERNAL3_6TB/TCGA_data/clinical_data/LUSC/LUSC_clinical_data_firebrowse_20210125.txt'

        df_raw1 = pd.read_csv(url_clinical1, sep='\t', index_col=0)
        df_raw2 = pd.read_csv(url_clinical2, sep='\t', index_col=0)

        df_raw = df_raw1.append(df_raw2)  # 1026 patients

    ''' Inspecting Data
    # check types of cols and adapt them
    df_raw.info(verbose=True)

    # look at statistics
    nuller = df_raw.isnull().sum() / 1026
    # nuller = nuller > 0.40
    print(nuller)
    # df_raw.describe()
    #look at unique values
    for x in list(df_raw.columns):
        print(x)
        col = df_raw[x]
        print(col.unique())
    '''

    ''' Feature selection 
    x values: 
        dates into -> OS_time_days
        death or followup -> OS_event

    features:

    gender: 0 male 1 female

    ethnicity: 'not hispanic or latino' 'hispanic or latino' 24,13% missing

    histological_type: 12 types -> df_selected.histological_type.value_counts()
        'lung adenocarcinoma mixed subtype' 'lung papillary adenocarcinoma'
     'lung bronchioloalveolar carcinoma nonmucinous'
     'lung adenocarcinoma- not otherwise specified (nos)'
     'mucinous (colloid) carcinoma' 'lung signet ring adenocarcinoma'
     'lung acinar adenocarcinoma' 'lung bronchioloalveolar carcinoma mucinous'
     'lung micropapillary adenocarcinoma' 'lung mucinous adenocarcinoma'
     'lung clear cell adenocarcinoma'
     'lung solid pattern predominant adenocarcinoma'

    karnofsky_performance_score: 73% missing

    number_pack_years_smoked: 31,8 % -> of smoked 1 if not 0
        year_of_tobacco_smoking_onset: left out because float binary

    pathologic_stage: 0 if stage I or II; 1 if else

    pathology_M_stage: 0 if nan or mx (not accessible) m0 ; 1 if m1

    pathology_N_stage: 0 if nan nx n0; 1 if n1 n2 n3

    pathology_T_stage: 0 if tx t1 t2; 1 if t3 t4

    race: nan 'white' 'black or african american' 'asian' 'american indian or alaska native' 12% missing

    radiation_therapy: nan yes no 9% missing

    residual_tumor: 0 if nan rx r0; 1 if r1 r2

    years_to_birth: age? float

    # Kick

    cols:
    date_of_initial_pathologic_diagnosis, days_to_death, days_to_last_followup, days_to_last_known_alive, karnofsky_performance_score, vital_status, years_to_birth, ethnicity  
    for now: tumor_tissue_site, histological_type, race

    patients:
    LUSC: ['TCGA-50-5045', 'TCGA-55-6969', 'TCGA-67-3772', 'TCGA-67-4679', 'TCGA-69-7765', 'TCGA-69-8254'] missing 3 or more values

    LUAD: ['TCGA-37-3789', 'TCGA-34-7107']

    is pathologic_state missing fine?

    '''
    excluded = ['race', 'histological_type', 'tumor_tissue_site', 'date_of_initial_pathologic_diagnosis',
                'days_to_death', 'days_to_last_followup', 'days_to_last_known_alive', 'OS_time_days', 'OS_event',
                'karnofsky_performance_score', 'vital_status', 'year_of_tobacco_smoking_onset', 'years_to_birth',
                'ethnicity']
    df_selected = df_raw.drop(excluded, axis=1)

    # preprocess cols
    df_selected['gender'] = [0 if x == 'male' else 1 for x in df_selected.gender]
    df_selected['number_pack_years_smoked'] = [1 if x > 0.0 else 0 for x in df_selected.number_pack_years_smoked]
    df_selected['pathologic_stage'] = [1 if x in ['stage iiib', 'stage iiia', 'stage iii', 'stage iv', np.nan] else 0
                                       for x in df_selected.pathologic_stage]
    df_selected['pathology_T_stage'] = [0 if x in ['t2b', 't2a', 't2', 't1b', 't1a', 't1', 't0', 'tx', np.nan] else 1
                                        for x in df_selected.pathology_T_stage]
    df_selected['pathology_N_stage'] = [0 if x in ['n0', 'nx', np.nan] else 1 for x in df_selected.pathology_N_stage]
    df_selected['pathology_M_stage'] = [0 if x in ['m0', 'mx', np.nan] else 1 for x in df_selected.pathology_M_stage]
    df_selected['radiation_therapy'] = [1 if x == 'yes' else 0 for x in df_selected.radiation_therapy]
    df_selected['residual_tumor'] = [1 if x in ['r1', 'r2'] else 0 for x in df_selected.residual_tumor]

    # Get dummies GETS RID of NAN! due to categoricals! One hot encoding
    df_binary = pd.get_dummies(df_selected, prefix_sep='_', drop_first=True)

    # save to txt file
    # df_binary.to_csv(r'/media/administrator/INTERNAL3_6TB/TCGA_data/clinical_data/NSCLC/NSCLC_clinical_binary.txt',index=True, sep='\t')

    # control values
    df_y_all = df_raw[['OS_time_days', 'OS_event']]
    if dataset == 'LUAD':
        df_y_all['Entity'] = ['LUAD' for _ in range(len(df_raw))]
    if dataset == 'NSCLC':
        df_y_all['Entity'] = ['LUAD' for _ in range(len(df_raw1))] + ['LUSC' for _ in range(len(df_raw2))]

    return df_binary, df_y_all
