### Script to do transitional RT clustering
from ThalHiEEG import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

ROOT='/data/backed_up/shared/ThalHi_data/eeg_preproc/'
print(os.listdir(ROOT))

#where data should go
OUT='/home/kahwang/bsh/ThalHi_data/TFR/'

###### NOTES on data structure
# under each subject folder, look for the cue-epo.fif, which is the cue epoch for all included_subjects
# there is a metadata field in the epoch object.
# look for the following dataframe cols "Color": {blue, red}, "Shape": {Polygon, Circle}, "Texture": {Donot, Filled}, "Task":{Scene, Face}
# perfomrance, look for "rt", "trial_Corr"
# this is in order

####### Behavior stats
# trial types are in "Trial_type" in the metadata field

##### load data
### control EEG
### looks like no swap for EEG
included_subjects = ['128', '112', '108', '110', '120', '98', '86', '82', '115', '94', '76', '91', '80', '95', '121', '114', '125', '70',
'107', '111', '88', '113', '131', '130', '135', '140', '167', '145', '146', '138', '147', '176', '122', '118', '103', '142']

# start big datafame
cdf = pd.DataFrame()
for sub in included_subjects:
    if sub not in ['73','96','137', '143', '200', '201', 'pilot', 'Em', 'ITI_epochs']:# and (sub=='Em' or sub=='112' or sub=='82'):#or sub=='80'): # sub 73 is the file with the 1 missing event, 103 and  96 are noisy subs. 200 and 201 are patients.
        this_sub_path=ROOT+sub
        cue_e = mne.read_epochs(this_sub_path+'/cue-epo.fif')
        cue_e.metadata['Subject'] = sub
        cdf = cdf.append(cue_e.metadata)
cdf.loc[cdf['rt']=='none', 'rt'] = np.nan
cdf['rt'] = cdf['rt'].astype('float')
cdf.groupby(['sub', 'Trial_type']).mean().groupby(['Trial_type']).mean()


### Pateint EEG
patients = ['200', '203', '201', '4036', '4032'] #203 and 4032 have more data. Get 4041
pdf = pd.DataFrame()
for sub in patients:
    this_sub_path=ROOT+sub
    cue_e = mne.read_epochs(this_sub_path+'/cue-epo.fif')
    cue_e.metadata['Subject'] = sub
    pdf = pdf.append(cue_e.metadata)

# 4041
import glob
fns = glob.glob('/home/kahwang/RDSS/ThalHi_data/EEG_data/behavioral_data/4041_00*csv')
fns.sort()

df_4041 = pd.DataFrame()
for fn in fns:
    df_4041 = df_4041.append(pd.read_csv(fn))
    df_4041['Subject'] = '4041'
pdf = pdf.append(df_4041)

pdf.loc[pdf['rt']=='none', 'rt'] = np.nan
pdf['rt'] = pdf['rt'].astype('float')
pdf['trial_Corr'] = pdf['trial_Corr'].astype('float')
summary_pdf = pdf.groupby(['Subject', 'Trial_type']).mean()


### Control fMRI
# source /home/kahwang/RDSS/ThalHi_data/MRI_data/Behavioral_data

fns = glob.glob('/home/kahwang/RDSS/ThalHi_data/MRI_data/Behavioral_data/100*.csv')
fns.sort()
fdf = pd.DataFrame()
for s, fn in enumerate(fns):
    tdf = pd.read_csv(fn)
    tdf['Subject'] = s
    fdf = fdf.append(tdf)

fdf.loc[fdf['sub']=='^10^016', 'sub']=10016
fdf.loc[fdf['rt']=='none', 'rt'] = np.nan
fdf['rt'] = fdf['rt'].astype('float')
fdf['trial_Corr'] = fdf['trial_Corr'].astype('float')

# drop bad subjects
fdf = fdf.loc[fdf['sub']!=10011]
fdf = fdf.loc[fdf['sub']!=10012]
fdf = fdf.loc[fdf['sub']!=10015]
fdf = fdf.loc[fdf['sub']!=10026]

# fix response
fdf.loc[(fdf['sub']==10006) & (fdf['trial_Corr']==0)]['trial_Corr'] = -1
fdf.loc[(fdf['sub']==10006) & (fdf['trial_Corr']==1)]['trial_Corr'] = 0
fdf.loc[(fdf['sub']==10006) & (fdf['trial_Corr']==-1)]['trial_Corr'] = 1

fdf.groupby(['sub', 'Trial_type']).mean().groupby(['Trial_type']).mean()


### create 8 by 8 transitional RT matrix. It WILL NOT be assymetric?
# col is previous trial, row is current trial, each cell is RT

def rt_cluster(inputdf):
    Subjects = inputdf['sub'].astype('int').unique()
    mat = np.zeros((8,8))
    subN = np.zeros((8,8))
    for s in Subjects:
        df = inputdf.loc[inputdf['sub'] == s]
        df = df.reset_index()

        #orgaznie matrix by texture, shape, color, return indx
        def get_positions(x):
            return {
                ('Filled', 'Polygon', 'red'): 0,
                ('Filled', 'Polygon', 'blue'): 1,
                ('Filled', 'Circle', 'red'): 2,
                ('Filled', 'Circle', 'blue'): 3,
                ('Donut', 'Polygon', 'red'): 4,
                ('Donut', 'Polygon', 'blue'): 5,
                ('Donut', 'Circle', 'red'): 6,
                ('Donut', 'Circle', 'blue'): 7,
            }[x]

        ### looks like no swap for EEG. so one matrix for all subjects
        transitionRTs = np.zeros((8,8))
        trialN = np.zeros((8,8))
        for i in df.index:
            if i == 0:
                continue
            else:
                if df.loc[i, 'trial_Corr']!=-1: # only correct trials
                    if df.loc[i-1, 'trial_Corr']!=-1:  # exclude post error slowing trials
                        previous_condition = get_positions((df.loc[i-1, 'Texture'], df.loc[i-1, 'Shape'], df.loc[i-1, 'Color']))
                        current_condition = get_positions((df.loc[i, 'Texture'], df.loc[i, 'Shape'], df.loc[i, 'Color']))
                        diff_rt = df.loc[i,'rt'] - df.loc[i-1, 'rt']
                        transitionRTs[previous_condition, current_condition] = transitionRTs[previous_condition, current_condition] + diff_rt
                        trialN[previous_condition, current_condition] =  trialN[previous_condition, current_condition] + 1

            ### or instead of 8 individual cell, group it by {solid circle, solid square, red hollows, blue hollows}. Four by four matrix.
            # Decoding hiearchial task structre for EEG?

        smat = transitionRTs/trialN
        smat[np.isnan(smat)] = np.nanmean(smat)
        smat = (smat - np.nanmean(smat)) / np.nanstd(smat)
        #subM = np.ones((8,8))
        #subM[np.isnan(smat)]=0
        #subN = subN + subM
        #mat = mat + smat
        #stack subjects
        mat = np.hstack((mat,smat))
    #mat = mat / subN

    return mat

## cluster the transitional RT matrix to see what conditions group together. We can do kmeans or dendrogram, or modularity
mat = rt_cluster(fdf)

# modularity
# import bct
# mat = mat -np.min(mat)
# mat=bct.invert(mat)
# print(bct.modularity_dir(mat.T))

i = np.zeros(len(np.arange(2,8)))
sil = []
for ix, n in enumerate((np.arange(2,8))):
    k = KMeans(n_clusters=n)
    k.fit(mat)
    print(k.labels_)
    i[ix] = k.inertia_
    sil.append(silhouette_score(mat, k.labels_, metric = 'euclidean'))
plt.figure()
sns.lineplot(y=sil, x=np.arange(2,8))

#Lets create a dendrogram variable linkage is actually the algorithm #itself of hierarchical clustering and then in linkage we have to #specify on which data we apply and engage. This is X dataset
plt.figure()
sch.dendrogram(sch.linkage(mat, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('condition')
plt.ylabel('')
plt.show()



# end of line
