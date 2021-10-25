import pandas as pd
import numpy as np
import pickle
import sys
import os

raw_datadir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.isdir(raw_datadir):
    print(raw_datadir + ' does not exist')
    sys.exit()
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

symptom_cols = []
symptom_cols_dict = dict()
total_cols = []
standard3_cols = ['PATNO','EVENT_ID', 'INFODT']
shared_cols = []
shared_cols_dict = dict()
screening_cols = []
screening_cols_dict = dict()
baseline_cols = []
baseline_cols_dict = dict()
infodt_only_cols = []
eventid_only_cols = []

def average_by_patno_eventid(df):
    '''
    Averages column values for same patient and event
    '''
    df_standard3_cols = df[standard3_cols].drop_duplicates(subset=['PATNO','EVENT_ID'])
    del df['INFODT']
    agg_dict = dict()
    for col in df.columns:
        if col == 'PATNO' or col == 'EVENT_ID':
            continue
        col_agg_dict = dict()
        col_agg_dict[col] = np.nanmean
        agg_dict[col] = col_agg_dict
    mean_df = df.groupby(by=['PATNO','EVENT_ID']).agg(agg_dict)
    mean_df.columns = mean_df.columns.droplevel(0)
    mean_df = mean_df.reset_index()
    assert len(mean_df) == len(df_standard3_cols)
    mean_df = mean_df.merge(df_standard3_cols, on=['PATNO','EVENT_ID'], validate='one_to_one')
    return mean_df

def merge_dfs(collection_df, specific_df):
    '''
    Merge two pandas dataframes based on patient and event
    '''
    # if needed, first take average if any 2 rows in specific_df share the same PATNO and EVENT_ID (keep either of the INFODT)
    # INFODT column should use collection_df's first, then specific_df's
    if len(specific_df) != len(specific_df.drop_duplicates(subset=['PATNO','EVENT_ID'])):
        specific_df = average_by_patno_eventid(specific_df)
    collection_df = collection_df.merge(specific_df, on=['PATNO','EVENT_ID'], how='outer', copy=False, validate = 'one_to_one')
    infodt_y_df = collection_df[['INFODT_y']]
    infodt_y_df.rename(columns={'INFODT_y':'INFODT'}, inplace=True)
    del collection_df['INFODT_y']
    collection_df.rename(columns={'INFODT_x':'INFODT'}, inplace=True)
    infodt_y_df['INFODT'] = pd.to_datetime(infodt_y_df['INFODT'])
    collection_df['INFODT'] = pd.to_datetime(collection_df['INFODT'])
    collection_df.update(infodt_y_df, overwrite=False)
    return collection_df

# MDS-UPDRS III
raw_mds_updrs3_path = raw_datadir + 'MDS_UPDRS_Part_III.csv'
raw_mds_updrs3_df = pd.read_csv(raw_mds_updrs3_path)
raw_mds_updrs3_df['LAST_UPDATE'] = pd.to_datetime(raw_mds_updrs3_df['LAST_UPDATE'])
raw_mds_updrs3_df['INFODT'] = pd.to_datetime(raw_mds_updrs3_df['INFODT'])
raw_mds_updrs3_df = raw_mds_updrs3_df.sort_values(by=standard3_cols + ['LAST_UPDATE'])
raw_mds_updrs3_df = raw_mds_updrs3_df.drop_duplicates(subset={'PATNO','EVENT_ID','INFODT','PD_MED_USE','CMEDTM'}, keep='last')
raw_mds_updrs3_df.rename(columns={'PN3RIGRL': 'NP3RIGRL'}, inplace=True)
updrs3_symptom_cols = ['NP3SPCH', 'NP3FACXP', 'NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL', 'NP3FTAPR', \
                       'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 'NP3PRSPL', 'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', \
                       'NP3LGAGL', 'NP3RISNG', 'NP3GAIT', 'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR', 'NP3BRADY', 'NP3PTRMR', \
                       'NP3PTRML', 'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL', 'NP3RTALJ', \
                       'NP3RTCON']
updrs3_total_col = 'NUPDRS3'
raw_mds_updrs3_df[updrs3_total_col] = raw_mds_updrs3_df[updrs3_symptom_cols].sum(axis=1)
'''
4 cases:
1. untreated: PD_MED_USE = 0
2. maob: PD_MED_USE = 3 (MAO-B inhibited, can't be defined as on/off)
3. on: PD_MED_USE not 0 or 3 AND (CMEDTM is not NaN OR ON_OFF_DOSE is 2)
4: off: PD_MED_USE is not 0 or 3 AND CMEDTM is NaN and ON_OFF_DOSE is not 2
'''
# very few PD_MED_USE == NaN - those will be thrown out
raw_mds_updrs3_df = raw_mds_updrs3_df.dropna(subset=['PD_MED_USE'])
# 4 patient-visits have CMEDTM despite PD_MED_USE = 0 -> TODO: handle this correctly, drop for now
raw_mds_updrs3_df = raw_mds_updrs3_df.loc[~np.logical_and(raw_mds_updrs3_df['PD_MED_USE']==0, \
                                                          ~pd.isnull(raw_mds_updrs3_df['CMEDTM']))]
raw_mds_updrs3_df_untreated = raw_mds_updrs3_df.loc[raw_mds_updrs3_df['PD_MED_USE']==0][standard3_cols + updrs3_symptom_cols \
                                                                                        + [updrs3_total_col]]
raw_mds_updrs3_df_maob = raw_mds_updrs3_df.loc[raw_mds_updrs3_df['PD_MED_USE']==3][standard3_cols + updrs3_symptom_cols \
                                                                                   + [updrs3_total_col]]
remaining_mds_updrs3_df = raw_mds_updrs3_df.loc[np.logical_and(raw_mds_updrs3_df['PD_MED_USE']!=0, \
                                                               raw_mds_updrs3_df['PD_MED_USE']!=3)]
raw_mds_updrs3_df_on \
    = remaining_mds_updrs3_df.loc[np.logical_or(~pd.isnull(remaining_mds_updrs3_df['CMEDTM']), \
                                                remaining_mds_updrs3_df['ON_OFF_DOSE']==2)][standard3_cols \
                                                                                            + updrs3_symptom_cols \
                                                                                            + [updrs3_total_col]]
raw_mds_updrs3_df_off \
    = remaining_mds_updrs3_df.loc[np.logical_and(pd.isnull(remaining_mds_updrs3_df['CMEDTM']), \
                                                 remaining_mds_updrs3_df['ON_OFF_DOSE']!=2)][standard3_cols \
                                                                                             + updrs3_symptom_cols \
                                                                                             + [updrs3_total_col]]
untreated_columns_map = dict()
untreated_cols = []
off_columns_map = dict()
off_cols = []
on_columns_map = dict()
on_cols = []
maob_columns_map = dict()
maob_cols = []
mean_columns_map = dict()
for col in updrs3_symptom_cols + ['NUPDRS3']:
    untreated_columns_map[col] = col + '_untreated'
    untreated_cols.append(col + '_untreated')
    off_columns_map[col] = col + '_off'
    off_cols.append(col + '_off')
    on_columns_map[col] = col + '_on'
    on_cols.append(col + '_on')
    maob_columns_map[col] = col + '_maob'
    maob_cols.append(col + '_maob')
    mean_columns_map[col] = {col: 'mean'}
raw_mds_updrs3_df_untreated = raw_mds_updrs3_df_untreated[standard3_cols + updrs3_symptom_cols + ['NUPDRS3']]
raw_mds_updrs3_df_untreated = raw_mds_updrs3_df_untreated.groupby(by=standard3_cols).agg(mean_columns_map)
raw_mds_updrs3_df_untreated.columns = raw_mds_updrs3_df_untreated.columns.droplevel(0)
raw_mds_updrs3_df_untreated = raw_mds_updrs3_df_untreated.reset_index()
raw_mds_updrs3_restruc_df = raw_mds_updrs3_df_untreated.rename(columns=untreated_columns_map)
raw_mds_updrs3_df_on = raw_mds_updrs3_df_on[standard3_cols + updrs3_symptom_cols + ['NUPDRS3']]
raw_mds_updrs3_df_on = raw_mds_updrs3_df_on.groupby(by=standard3_cols).agg(mean_columns_map)
raw_mds_updrs3_df_on.columns = raw_mds_updrs3_df_on.columns.droplevel(0)
raw_mds_updrs3_df_on = raw_mds_updrs3_df_on.reset_index()
raw_mds_updrs3_df_on.rename(columns=on_columns_map, inplace=True)
raw_mds_updrs3_df_off = raw_mds_updrs3_df_off[standard3_cols + updrs3_symptom_cols + ['NUPDRS3']]
raw_mds_updrs3_df_off = raw_mds_updrs3_df_off.groupby(by=standard3_cols).agg(mean_columns_map)
raw_mds_updrs3_df_off.columns = raw_mds_updrs3_df_off.columns.droplevel(0)
raw_mds_updrs3_df_off = raw_mds_updrs3_df_off.reset_index()
raw_mds_updrs3_df_off.rename(columns=off_columns_map, inplace=True)
raw_mds_updrs3_df_maob = raw_mds_updrs3_df_maob[standard3_cols + updrs3_symptom_cols + ['NUPDRS3']]
raw_mds_updrs3_df_maob = raw_mds_updrs3_df_maob.groupby(by=standard3_cols).agg(mean_columns_map)
raw_mds_updrs3_df_maob.columns = raw_mds_updrs3_df_maob.columns.droplevel(0)
raw_mds_updrs3_df_maob = raw_mds_updrs3_df_maob.reset_index()
raw_mds_updrs3_df_maob.rename(columns=maob_columns_map, inplace=True)
raw_mds_updrs3_restruc_df = raw_mds_updrs3_restruc_df.merge(raw_mds_updrs3_df_off, how = 'outer', validate = 'one_to_one')
raw_mds_updrs3_restruc_df = raw_mds_updrs3_restruc_df.merge(raw_mds_updrs3_df_on, how = 'outer', validate = 'one_to_one')
raw_mds_updrs3_restruc_df = raw_mds_updrs3_restruc_df.merge(raw_mds_updrs3_df_maob, how = 'outer', validate = 'one_to_one')
for col in raw_mds_updrs3_restruc_df:
    if col not in standard3_cols:
        raw_mds_updrs3_restruc_df[col] = raw_mds_updrs3_restruc_df[col].astype(np.float64)
if len(raw_mds_updrs3_restruc_df) != len(raw_mds_updrs3_restruc_df.drop_duplicates(subset=['PATNO','EVENT_ID'])):
    raw_mds_updrs3_restruc_df = average_by_patno_eventid(raw_mds_updrs3_restruc_df)
updrs3_total_cols = [untreated_cols[-1], off_cols[-1], on_cols[-1], maob_cols[-1]]
updrs3_symptom_cols = untreated_cols[:-1] + off_cols[:-1] + on_cols[:-1] + maob_cols[:-1]
symptom_cols += updrs3_symptom_cols
symptom_cols_dict['NUPDRS3'] = updrs3_symptom_cols
total_cols += updrs3_total_cols

# MDS-UPDRS II
raw_mds_updrs2_path = raw_datadir + 'MDS_UPDRS_Part_II__Patient_Questionnaire.csv'
raw_mds_updrs2_df = pd.read_csv(raw_mds_updrs2_path)
updrs2_symptom_cols = ['NP2SPCH', 'NP2SALV', 'NP2SWAL', 'NP2EAT', 'NP2DRES', 'NP2HYGN', 'NP2HWRT', 'NP2HOBB', 'NP2TURN', \
                       'NP2TRMR', 'NP2RISE', 'NP2WALK','NP2FREZ']
updrs2_total_col = 'NUPDRS2'
raw_mds_updrs2_df[updrs2_total_col] = raw_mds_updrs2_df[updrs2_symptom_cols].sum(axis=1)
raw_mds_updrs2_df = raw_mds_updrs2_df[standard3_cols+updrs2_symptom_cols+[updrs2_total_col]]
for col in raw_mds_updrs2_df:
    if col not in standard3_cols:
        raw_mds_updrs2_df[col] = raw_mds_updrs2_df[col].astype(np.float64)
symptom_cols += updrs2_symptom_cols
symptom_cols_dict['NUPDRS2'] = updrs2_symptom_cols
total_cols.append(updrs2_total_col)
print('Merging MDS-UPDRS III + II')
collected_data = merge_dfs(raw_mds_updrs3_restruc_df, raw_mds_updrs2_df)

# Subject center number
cno_path = raw_datadir + 'Center-Subject_List.csv'
cno_df = pd.read_csv(cno_path)
collected_data = collected_data.merge(cno_df[['PATNO','CNO']],on='PATNO',how='outer',validate='many_to_one')
collected_data['CNO'] = collected_data['CNO'].astype(np.float64)
baseline_cols.append('CNO')
baseline_cols_dict['CNO'] = ['CNO']

# Age
random_path = raw_datadir + 'Randomization_table.csv'
random_df = pd.read_csv(random_path)
collected_data = collected_data.merge(random_df[['PATNO','BIRTHDT']], on='PATNO', how='outer')

# Gender
collected_data = collected_data.merge(random_df[['PATNO','GENDER']], on='PATNO', how='outer')
collected_data['MALE'] = np.where(collected_data['GENDER']==2, 1, 0)
#collected_data['FEMALE'] = np.where(collected_data['GENDER']==2, 0, 1)
del collected_data['GENDER']
gender_cols = ['MALE']#, 'FEMALE']
baseline_cols += gender_cols
baseline_cols_dict['GENDER'] = gender_cols
for col in gender_cols:
    collected_data[col] = collected_data[col].astype(np.float64)

# calculate age, disease duration, and time since baseline visit at each visit
collected_data['AGE'] = (pd.to_datetime(collected_data['INFODT']) - pd.to_datetime(collected_data['BIRTHDT']))/ np.timedelta64(1, 'Y')
del collected_data['BIRTHDT']
shared_cols.append('AGE')
shared_cols_dict['AGE'] = ['AGE']
collected_data['INFODT_DIS_DUR'] = (pd.to_datetime(collected_data['INFODT']) \
                                    - pd.to_datetime(collected_data['PDDXDT']))/ np.timedelta64(1, 'Y')
collected_data['INFODT_TIME_SINCE_ENROLL'] = (pd.to_datetime(collected_data['INFODT']) - pd.to_datetime(collected_data['ENROLL_DATE']))/ np.timedelta64(1, 'Y')
collected_data['INFODT_TIME_SINCE_ENROLL'] = collected_data['INFODT_TIME_SINCE_ENROLL'].astype(np.float64)
del collected_data['ENROLL_DATE']

# convert ST to its appropriate visit
st_catalog_path = raw_datadir + 'ST_CATALOG.csv'
st_catalog_df = pd.read_csv(st_catalog_path)
collected_data = collected_data.merge(st_catalog_df[['PATNO','STRPLCVS']], on=['PATNO'], how='left', validate='many_to_one')
collected_data['EVENT_ID'] = np.where(np.logical_and(collected_data['EVENT_ID']=='ST', \
                                                     ~pd.isnull(collected_data['STRPLCVS'])), \
                                      collected_data['STRPLCVS'], collected_data['EVENT_ID'])
del collected_data['STRPLCVS']
# Drop unscheduled visits starting with U since little data taken at unscheduled visits anyways
collected_data = collected_data.loc[~collected_data['EVENT_ID'].str.startswith('U', na=False)]

event_id_dur_dict = {'SC': 0, 'BL': 1.5, 'V01': 4.5, 'V02': 7.5, 'V03': 10.5, 'V04': 13.5, 'V05': 19.5, 'V06': 25.5, \
                     'V07': 31.5, 'V08': 37.5, 'V09': 43.5, 'V10': 49.5, 'V11': 55.5, 'V12': 61.5, 'V13': 73.5, 'V14': 85.5, \
                     'V15': 97.5, 'BSL': 1.5, 'PV02': 7.5, 'PV04': 13.5, 'PV05': 19.5, 'PV06': 25.5, \
                     'PV07': 31.5, 'PV08': 37.5, 'PV09': 43.5, 'PV10': 49.5, 'PV11': 55.5, 'PV12': 61.5, \
                     'P13': 79.5, 'P14': 91.5, 'P15': 103.5, 'V16': 109.5, 'P16': 115.5, 'V17': 121.5, 'P17': 127.5, \
                     'V18': 133.5, 'P18': 139.5, 'V19': 145.5, 'P19': 151.5, 'V20': 157.5}
collected_data['EVENT_ID_DUR'] = collected_data['INFODT_TIME_SINCE_ENROLL']
for event_id in event_id_dur_dict:
    collected_data['EVENT_ID_DUR'] = np.where(collected_data['EVENT_ID']==event_id, event_id_dur_dict[event_id]/12., \
                                              collected_data['EVENT_ID_DUR'])
collected_data['EVENT_ID_DUR'] = collected_data['EVENT_ID_DUR'].astype(np.float64)
collected_data['DIS_DUR_BY_CONSENTDT'] = (pd.to_datetime(collected_data['CONSNTDT']) \
                                          - pd.to_datetime(collected_data['PDDXDT']))/ np.timedelta64(1, 'Y') \
                                        + collected_data['EVENT_ID_DUR']
del collected_data['CONSNTDT']
del collected_data['PDDXDT']
shared_cols.remove('PDDXDT')
time_feats = ['INFODT_DIS_DUR', 'INFODT_TIME_SINCE_ENROLL', 'EVENT_ID_DUR', 'DIS_DUR_BY_CONSENTDT']
screening_cols += time_feats
screening_cols_dict['TIME_FEATS'] = time_feats
baseline_cols += time_feats
baseline_cols_dict['TIME_FEATS'] = time_feats
total_cols += time_feats
symptom_cols += time_feats
symptom_cols_dict['TIME_FEATS'] = time_feats
shared_cols += time_feats
shared_cols_dict['TIME_FEATS'] = time_feats

binary_cols = set()
all_cols = screening_cols + baseline_cols + symptom_cols + total_cols + shared_cols
for col in all_cols:
    col_value_counts = collected_data[[col]].dropna()[col].value_counts()
    if len(col_value_counts) <= 2 and set(col_value_counts.keys()).issubset(set({0,1})):
        binary_cols.add(col)

def output_baseline_features(df, cols):
    '''
    Prints aggregate statistics of static features
    '''
    output_str = ''
    agg_stats = []
    agg_stat_names = []
    for feat in cols:
        if feat == 'CNO':
            num_cnos = df.CNO.nunique()
            agg_stats.append(num_cnos)
            agg_stat_names.append('CNO num sites')
            continue
        nonnan_feat_df = df.loc[~pd.isnull(df[feat])][['PATNO',feat]]
        if feat in binary_cols:
            if len(nonnan_feat_df) == 0:
                binary_freq1 = 0.
                feat_num_patnos = 0
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                feat_vals = nonnan_feat_df[feat].value_counts()
                if 1 in feat_vals.keys():
                    binary_freq1 = feat_vals[1]/float(len(nonnan_feat_df))
                else:
                    binary_freq1 = 0.
            output_str += feat + ': ' + str(binary_freq1) + ', ' + str(feat_num_patnos) + '\n'
            agg_stats += [binary_freq1, feat_num_patnos]
            agg_stat_names += [feat + '_freq', feat + '_num_patnos']
        else:
            if len(nonnan_feat_df) == 0:
                feat_10 = 0.
                feat_mean = 0.
                feat_90 = 0.
                feat_num_patnos = 0.
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                feat_arr = nonnan_feat_df[feat].values.astype(np.float64)
                nonnan_feat_df_noinf = nonnan_feat_df.loc[nonnan_feat_df[feat]!=float('+inf')]
                nonnan_feat_df_noinf = nonnan_feat_df_noinf.loc[nonnan_feat_df_noinf[feat]!=float('-inf')]
                feat_arr_noinf = nonnan_feat_df_noinf[feat].values.astype(np.float64)
                feat_10 = np.percentile(feat_arr, 10)
                feat_mean = np.mean(feat_arr_noinf)
                feat_90 = np.percentile(feat_arr, 90)
            output_str += feat + ': ' + str(feat_10) + ', ' + str(feat_mean) + ', ' + str(feat_90) + ', ' \
                + str(feat_num_patnos) + '\n'
            agg_stats += [feat_10, feat_mean, feat_90, feat_num_patnos]
            agg_stat_names += [feat + '_10', feat + '_mean', feat + '_90', feat + '_num_patnos']
    return output_str, agg_stats, agg_stat_names

def output_changing_features(df, cols):
    '''
    Prints aggregate statistics of changing features
    '''
    output_str = ''
    agg_stats = []
    agg_stat_names = []
    for feat in cols:
        nonnan_feat_df = df.loc[~pd.isnull(df[feat])]
        if feat in binary_cols:
            if len(nonnan_feat_df) == 0:
                binary_freq1 = 0.
                feat_num_patnos = 0
                avg_num_visits = 0.
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                avg_num_visits = len(nonnan_feat_df)/float(feat_num_patnos)
                feat_vals = nonnan_feat_df[feat].value_counts()
                if 1 in feat_vals.keys():
                    binary_freq1 = feat_vals[1]/float(len(nonnan_feat_df))
                else:
                    binary_freq1 = 0
            output_str += feat + ': ' + str(binary_freq1) + ', ' + str(feat_num_patnos) + ', ' + str(avg_num_visits) + '\n'
            agg_stats += [binary_freq1, feat_num_patnos, avg_num_visits]
            agg_stat_names += [feat + '_freq', feat + '_num_patnos', feat + '_avg_num_visits']
        else:
            if len(nonnan_feat_df) == 0:
                feat_10 = 0.
                feat_mean = 0.
                feat_90 = 0.
                feat_num_patnos = 0
                avg_num_visits = 0.
            else:
                feat_num_patnos = nonnan_feat_df.PATNO.nunique()
                avg_num_visits = len(nonnan_feat_df)/float(feat_num_patnos)
                feat_vals = nonnan_feat_df[feat].value_counts()
                feat_arr = nonnan_feat_df[feat].values.astype(np.float64)
                nonnan_feat_df_noinf = nonnan_feat_df.loc[nonnan_feat_df[feat]!=float('+inf')]
                nonnan_feat_df_noinf = nonnan_feat_df_noinf.loc[nonnan_feat_df_noinf[feat]!=float('-inf')]
                feat_arr_noinf = nonnan_feat_df_noinf[feat].values.astype(np.float64)
                feat_10 = np.percentile(feat_arr, 10)
                feat_mean = np.mean(feat_arr_noinf)
                feat_90 = np.percentile(feat_arr, 90)
            output_str += feat + ': ' + str(feat_10) + ', ' + str(feat_mean) + ', ' + str(feat_90) + ', ' \
                + str(feat_num_patnos) + ', ' + str(avg_num_visits) + '\n'
            agg_stats += [feat_10, feat_mean, feat_90, feat_num_patnos, avg_num_visits]
            agg_stat_names += [feat + '_10', feat + '_mean', feat + '_90', feat + '_num_patnos', feat + '_avg_num_visits']
    return output_str, agg_stats, agg_stat_names

def output_df(df, cohort_name):
    '''
    write 4 csvs per dataframe: 3 across time (symptoms, totals, shared), 1 for baseline
    summary stats for dataframe to csv
    '''
    total_filename = output_dir + cohort_name + '_totals_across_time.csv'
    total_df = df[standard3_cols + total_cols]
    total_df.to_csv(total_filename, index=False)
    shared_filename = output_dir + cohort_name + '_other_across_time.csv'
    shared_df = df[standard3_cols + shared_cols]
    shared_df.to_csv(shared_filename, index=False)
    screening_filename = output_dir + cohort_name + '_screening.csv'
    baseline_filename = output_dir + cohort_name + '_baseline.csv'
    sc_baseline_df = df.loc[df['EVENT_ID']=='SC'][standard3_cols + screening_cols]
    assert len(sc_baseline_df) == sc_baseline_df.PATNO.nunique()
    sc_baseline_df.to_csv(screening_filename, index=False)
    bl_baseline_df = df.loc[df['EVENT_ID']=='BL'][standard3_cols + baseline_cols]
    bl_missing_patnos = set(df.PATNO.unique()).difference(set(bl_baseline_df.PATNO.unique()))
    bl_missing_patnos_df = df.loc[df['PATNO'].isin(bl_missing_patnos)][standard3_cols + baseline_cols].dropna(how='all')
    bl_missing_patnos_df = bl_missing_patnos_df.sort_values(by=['INFODT'])
    bl_missing_patnos_df = bl_missing_patnos_df.drop_duplicates(subset=['PATNO'], keep='first')
    bl_baseline_df = pd.concat([bl_baseline_df, bl_missing_patnos_df])
    assert len(bl_baseline_df) == bl_baseline_df.PATNO.nunique()
    bl_baseline_df.to_csv(baseline_filename, index=False)
    
    output_str = cohort_name + '\n'
    output_str += str(df.PATNO.nunique()) + ' patients\n'
    agg_stats = [df.PATNO.nunique()]
    agg_stat_names = ['# patients']
    output_str += str(len(screening_cols)) + ' screening features, ' + str(len(baseline_cols)) + ' baseline features, ' \
        + str(len(total_cols)) + ' assessment totals across time, ' + str(len(symptom_cols)) \
        + ' assessment questions across time, and ' + str(len(shared_cols)) + ' other features across time\n'
    
    output_str += 'Screening features: binary frequency or {10th percentile, mean, 90th percentile}, # patients\n'
    screening_output_str, screening_agg_stats, screening_agg_stat_names = output_baseline_features(sc_baseline_df, screening_cols)
    output_str += screening_output_str
    agg_stats += screening_agg_stats
    agg_stat_names += screening_agg_stat_names
    
    output_str += 'Baseline features: binary frequency or {10th percentile, mean, 90th percentile}, # patients\n'
    baseline_output_str, baseline_agg_stats, baseline_agg_stat_names = output_baseline_features(bl_baseline_df, baseline_cols)
    output_str += baseline_output_str
    agg_stats += baseline_agg_stats
    agg_stat_names += baseline_agg_stat_names
    
    output_str += 'Assessment totals across time: binary frequency or {10th percentile, mean, 90th percentile}, # patients, avg # visits per patient\n'
    total_output_str, total_agg_stats, total_agg_stat_names = output_changing_features(total_df, total_cols)
    output_str += total_output_str
    agg_stats += total_agg_stats
    agg_stat_names += total_agg_stat_names
    
    output_str += 'Assessment questions across time: binary frequency or {10th percentile, mean, 90th percentile}, # patients, avg # visits per patient\n'
    symptom_output_str, symptom_agg_stats, symptom_agg_stat_names = output_changing_features(symptom_df, symptom_cols)
    output_str += symptom_output_str
    agg_stats += symptom_agg_stats
    agg_stat_names += symptom_agg_stat_names
    
    output_str += 'Other features across time: binary frequency or {10th percentile, mean, 90th percentile}, # patients, avg # visits per patient\n'
    other_output_str, other_agg_stats, other_agg_stat_names = output_changing_features(shared_df, shared_cols)
    output_str += other_output_str
    agg_stats += other_agg_stats
    agg_stat_names += other_agg_stat_names
    
    agg_stat_df = pd.DataFrame(agg_stat_names, columns=['Stats'])
    agg_stat_df[cohort_name] = agg_stats
    
    summ_stat_file = output_dir + cohort_name + '_summary.txt'
    with open(summ_stat_file, 'w') as f:
        f.write(output_str)
    
    return agg_stat_df

# include only patients in de novo PD cohort
patient_status_path = raw_datadir + 'Patient_Status.csv'
patient_status_df = pd.read_csv(patient_status_path)
enrolled_patient_status_df = patient_status_df.loc[~patient_status_df['ENROLL_DATE'].isnull()][['PATNO','ENROLL_CAT']]
pd_cohort_df = enrolled_patient_status_df.loc[enrolled_patient_status_df['ENROLL_CAT']=='PD']

agg_stat_df = output_df(collected_data.loc[collected_data['PATNO'].isin(pd_cohort_df.PATNO.unique())], 'PD')
agg_stat_df.to_csv(output_dir + 'agg_stat.csv', index=False)