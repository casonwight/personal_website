import pandas as pd
import os
import numpy as np
from plotnine import ggplot, aes, coord_flip, geom_bar, labs, geom_line, facet_wrap, geom_point, theme_light, theme, element_blank
import shelve
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import datetime
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import re, seaborn as sns
import pylab

os.chdir('C:\\Users\\cason\\Desktop\\Classes\\Assignments\\Stat 666\\Final Project')

my_shelf = shelve.open('Feature_Selection_Results.out')
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()


overall_mean = dfwords.mean()
dfwords['Speaker'] = X_train_all.Speaker
oaks = dfwords.query('Speaker == "Dallin H. Oaks"').mean()
oaks = (oaks - overall_mean).reset_index()
oaks.columns = ['Word', 'Mean TF-IDF Score']
oaks = oaks.sort_values('Mean TF-IDF Score', ascending = False).iloc[[0,2,3,5,6,2496,2497,2498,2499],:]
oaks.Word = pd.Categorical(oaks.Word, categories = oaks.sort_values('Mean TF-IDF Score')['Word'])

oaks_unique = (ggplot(oaks, aes(x = 'Word', y = 'Mean TF-IDF Score'))
     + geom_bar(stat = 'identity')
     + coord_flip()
     + labs(title = "Elder Oaks' Most unique Vocabulary (Included/Excluded)", 
            y = "TF-IDF Score: Oaks' Mean - Overall Mean"))
oaks_unique.save('oaks-unique-plot.png', width = 12, height = 6, dpi = 1200)


interesting_words = ['book mormon', 'mormon', 'holy ghost', 'prophet joseph smith', 'ministering', 'bear testimony']


most_used = dfwords.mean().sort_values(ascending = False)[interesting_words].reset_index()
most_used.columns = ['Word', 'Mean TF-IDF Score']
most_used.Word = pd.Categorical(most_used.Word, categories = most_used.sort_values('Mean TF-IDF Score')['Word'])

(ggplot(most_used, aes(x = 'Word', y = 'Mean TF-IDF Score'))
     + geom_bar(stat = 'identity')
     + coord_flip())

first_presidency = list(all_talks.Speaker.unique())[:3]
first_pres_num = [speaker_dict[mbr] for mbr in first_presidency]

dfwords['Speaker'] = y_train
by_speaker = (dfwords
              .groupby('Speaker')
              .mean()[most_used.sort_values('Mean TF-IDF Score')['Word']]
              .unstack()
              .reset_index()
              .query(f'Speaker in {first_pres_num}'))
by_speaker.columns = ['Word', 'Speaker', 'Mean TF-IDF Score']
by_speaker.Speaker = [to_speaker_dict[spkr] for spkr in by_speaker.Speaker]
by_speaker.Word = pd.Categorical(by_speaker.Word, categories = most_used.Word.sort_index(ascending=False))
by_speaker = by_speaker.append(most_used)
by_speaker.iloc[list(np.where(by_speaker.Speaker.isnull())[0]),1] = 'Overall'
by_speaker.Speaker = pd.Categorical(by_speaker.Speaker, categories = first_presidency + ['Overall'])
first_pres_words = (ggplot(by_speaker, aes(x = 'Word', y = 'Mean TF-IDF Score', fill = 'Speaker', width = .75))
     + geom_bar(position = 'dodge', stat = 'identity')
     + coord_flip()
     + labs(title = "Word Usage", x = "Phrase")
     + theme_light()
    + theme(legend_position = "bottom", legend_background = element_blank()))
first_pres_words.save('first-pres-words-plot.png', width = 12, height = 5, dpi = 1200)


dfwords['Date'] = X_train_all.Date.values
dfwords.Speaker = [to_speaker_dict[spkr] for spkr in dfwords.Speaker]
pres_Nelson = (dfwords
               .query(f"Speaker in {first_presidency}")
               .groupby([dfwords['Date']
               .map(lambda x: x.year), 'Speaker'])
               .mean()[interesting_words]
               .reset_index())
pres_Nelson['combined'] = [str(dt) for dt in pres_Nelson.Date] + pres_Nelson.Speaker
pres_Nelson = (pres_Nelson
               .drop(columns = ['Date', 'Speaker'])
               .set_index('combined')
               .unstack()
               .reset_index())
pres_Nelson['Date'] = [int(comb[:4]) for comb in pres_Nelson.combined]
pres_Nelson['Speaker'] = [comb[4:] for comb in pres_Nelson.combined]
pres_Nelson = pres_Nelson.drop(columns = 'combined')
pres_Nelson.columns = ['Word', 'Mean TF-IDF Score', 'Date', 'Speaker']

first_pres_time = (ggplot(pres_Nelson, aes(x = 'Date', y = 'Mean TF-IDF Score', color = 'Word'))
     + geom_line()
     + facet_wrap('Speaker', scales = 'free', nrow = 3)
     + labs(title = 'Word Usage over Time (First Presidency)', color = "Phrase")
     + theme_light()
     + theme(legend_position = "bottom", legend_background = element_blank()))
first_pres_time.save('first-pres-time-plot.png', width = 12, height = 5, dpi = 1200)


int_words = (dfwords
               .groupby(dfwords['Date']
               .map(lambda x: x.year))
               .mean()[interesting_words]
               .unstack()
               .reset_index())
int_words.columns = ['Word', 'Date', 'Mean TF-IDF Score']
(ggplot(int_words, aes(x = 'Date', y = 'Mean TF-IDF Score', color = 'Word'))
    + geom_line()
    + labs(title = 'Word Usage Change over Time in First Presidency and the 12'))

missionary_temple = (dfwords
               .groupby(dfwords['Date']
               .map(lambda x: x.year))
               .mean()[['missionary work', 'family history']]
               .unstack()
               .reset_index())
missionary_temple.columns = ['Word', 'Date', 'Mean TF-IDF Score']
missionary_temple_plot = (ggplot(missionary_temple, aes(x = 'Date', y = 'Mean TF-IDF Score', color = 'Word'))
    + geom_line()
    + labs(title = 'Word Usage over Time', color = "Phrase")
    + theme_light()
    + theme(legend_position = "bottom", legend_background = element_blank()))
missionary_temple_plot.save('missionary-temple-plot.png', width = 12, height = 5, dpi = 1200)

youth = (dfwords
               .groupby(dfwords['Date']
               .map(lambda x: x.year))
               .mean()[['young men', 'young women']]
               .unstack()
               .reset_index())
youth.columns = ['Word', 'Date', 'Mean TF-IDF Score']
youth_plot = (ggplot(youth, aes(x = 'Date', y = 'Mean TF-IDF Score', color = 'Word'))
    + geom_line()
    + labs(title = 'Word Usage over Time', color = "Phrase")
    + theme_light()
    + theme(legend_position = "bottom", legend_background = element_blank()))
youth_plot.save('youth-plot.png', width = 12, height = 5, dpi = 1200)


scripture = (dfwords
               .groupby(dfwords['Date']
               .map(lambda x: x.year))
               .mean()[['book mormon', 'bible']]
               .unstack()
               .reset_index())
scripture.columns = ['Word', 'Date', 'Mean TF-IDF Score']
scripture_plot = (ggplot(scripture, aes(x = 'Date', y = 'Mean TF-IDF Score', color = 'Word'))
    + geom_line()
    + labs(title = 'Word Usage over Time', color = "Phrase")
    + theme_light()
    + theme(legend_position = "bottom", legend_background = element_blank()))
scripture_plot.save('scripture-plot.png', width = 12, height = 5, dpi = 1200)



pca = PCA(n_components=3)
pca_df = pca.fit_transform(tfidf_X_train.todense())

lda = LinearDiscriminantAnalysis(n_components = 3)
lda_df = lda.fit_transform(tfidf_X_train.todense(), y_train)

principalDf = pd.DataFrame(data = pca_df, columns = ['pc1', 'pc2', 'pc3'])
principalDf['Speaker_num'] = y_train
recent_Oaks = list(np.where([X_train_all.Date[i] > datetime.datetime(2020,1,1) and X_train_all.Speaker[i] == 'Dallin H. Oaks' for i in X_train_all.index])[0])
principalDf['Speaker'] = [to_speaker_dict[y_val] for y_val in y_train]
principalDf.loc[recent_Oaks, 'Speaker'] = '2020 Dallin H. Oaks'
principalDf.loc[recent_Oaks, 'Speaker_num'] = 15

linearDF = pd.DataFrame(data = lda_df, columns = ['lda1', 'lda2', 'lda3'])
linearDF['Speaker_num'] = y_train
linearDF['Speaker'] = [to_speaker_dict[y_val] for y_val in y_train]
linearDF.loc[recent_Oaks, 'Speaker'] = '2020 Dallin H. Oaks'
linearDF.loc[recent_Oaks, 'Speaker_num'] = 15

first_presidency = list(all_talks['Speaker'].unique()[:3]) + ['2020 Dallin H. Oaks']
first_pres_pca_df = principalDf.query(f'Speaker in {first_presidency}')
first_pres_lda_df = linearDF.query(f'Speaker in {first_presidency}')


colors_dict = {0:'red', 
               1:'lightblue', 
               2:'black', 
               3:'orange',
               4:'brown',
               5:'tab:orange',
               6:'tab:purple',
               7:'tab:green',
               8:'tab:pink',
               9:'tab:olive',
               10:'tab:gray',
               11:'tab:cyan',
               12:'b',
               13:'r',
               14:'darkslategray',
               15: 'blue'}
principalDf['plot_color'] = principalDf.Speaker_num.map(colors_dict)
linearDF['plot_color'] = linearDF.Speaker_num.map(colors_dict)


cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

fig = plt.figure()
ax =fig.gca(projection='3d')
for speaker in principalDf.Speaker.unique():
    ax.scatter(principalDf.query(f'Speaker == "{speaker}"').pc1, 
            principalDf.query(f'Speaker == "{speaker}"').pc2,
            principalDf.query(f'Speaker == "{speaker}"').pc3,
            label = speaker, marker = 'o', s = .6)

#ax.scatter(principalDf.pc1, principalDf.pc2, principalDf.pc3, zdir='z', c=principalDf.Speaker, 
#           depthshade=False, s=1, marker='.', cmap = cmap)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
#ax.legend()

ax.view_init(azim = 6)

def rotate(angle):
    ax.view_init(azim=angle)
    
rot_animation = animation.FuncAnimation(fig, rotate, 
                                        frames=np.arange(0, 362, 2), 
                                        interval=100)
rot_animation.save('3D_Principal_Components.gif', dpi=80)


fig = plt.figure()
ax =fig.gca(projection='3d')
for speaker in linearDF.Speaker.unique():
    ax.scatter(linearDF.query(f'Speaker == "{speaker}"').lda1, 
            linearDF.query(f'Speaker == "{speaker}"').lda2,
            linearDF.query(f'Speaker == "{speaker}"').lda3,
            label = speaker, marker = 'o', s = 20)

#ax.scatter(principalDf.pc1, principalDf.pc2, principalDf.pc3, zdir='z', c=principalDf.Speaker, 
#           depthshade=False, s=1, marker='.', cmap = cmap)
ax.set_xlabel('LDA 1')
ax.set_ylabel('LDA 2')
ax.set_zlabel('LDA 3')


ax.view_init(azim = 6)

rot_animation = animation.FuncAnimation(fig, rotate, 
                                        frames=np.arange(0, 362, 2), 
                                        interval=100)
rot_animation.save('3D_Linear.gif', dpi=80)


