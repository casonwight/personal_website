import sys
import requests
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc


##############################################################################
#           Scraping General Conference Talks

church_URL = 'https://www.churchofjesuschrist.org/'

escapes = ''.join([chr(char) for char in range(1, 32)])
translator = str.maketrans('', '', escapes)

all_speakers_URL = 'general-conference/speakers?lang=eng'

all_speakers_page = requests.get(church_URL + all_speakers_URL)
all_speaker_soup = BeautifulSoup(all_speakers_page.content, 'lxml')

all_speaker_links_soup = all_speaker_soup.find_all('div', "lumen-tile__title")
all_links = [spkr.find('a')['href'][1:] for spkr in all_speaker_links_soup]
links_to_use = all_links[:15]

page_num = str(1)

talks = pd.DataFrame(columns = ['Speaker', 'Date', 'Title', 'URL'])

for speaker_link in links_to_use:
    page = requests.get(church_URL + speaker_link)
    
    scripture_soup = BeautifulSoup(page.content, 'lxml')
    all_links = scripture_soup.find_all('a', 'pages-nav__list-item__anchor')
    if len(all_links) > 0:
        link_texts = [int(link.text) if len(link.text) == 1 else 0 
                      for link in all_links]
        num_pages = np.max(link_texts)
    else:
        num_pages = 1

    for this_page_num in range(1,num_pages+1):
        this_URL = church_URL + speaker_link + "&page=" + str(this_page_num)
        this_page = requests.get(this_URL)
        speaker_soup = BeautifulSoup(this_page.content, 'lxml')
        
        this_speaker = speaker_soup.find('h1').get_text()
        
        talk_titles_soup = speaker_soup.find_all('div', "lumen-tile__title")
        these_titles = [title.get_text().translate(translator) 
                        for title in talk_titles_soup]
    
        talk_dates_soup = speaker_soup.find_all('div', "lumen-tile__content")
        these_dates_str = [date.get_text().translate(translator) 
                           for date in talk_dates_soup]
        these_dates = [datetime.strptime(date_str, "%B %Y") 
                       for date_str in these_dates_str]
    
        talk_links_soup = speaker_soup.find_all('a', "lumen-tile__link")
        these_URLs = [church_URL + link['href'][1:] 
                      for link in talk_links_soup]
        
        these_talks = pd.DataFrame({
                        'Speaker':[this_speaker for i in these_URLs],
                        'Date':these_dates,
                        'Title':these_titles,
                        'URL':these_URLs})
        
        talks = talks.append(these_talks, ignore_index = True)

all_content = []

for talk_row in range(np.shape(talks)[0]):
    perc_comp = talk_row / np.shape(talks)[0] * 100
    sys.stdout.write(f"\rWebscrape Progress: {int(np.round(perc_comp))}%")
    talk_URL = talks['URL'][talk_row]
    talk_page = requests.get(talk_URL)
    talk_soup = BeautifulSoup(talk_page.content, 'lxml')
    ids = [p.get('id') for p in talk_soup.find_all('p')]
    paragraphs = [idx[0] == 'p' and idx[1].isdigit() if idx != None else False 
                  for idx in ids]
    paragraph_ids = [ids[paragraph_index] 
                     for paragraph_index in list(np.where(paragraphs)[0])]
    
    content = [talk_soup.find(id = this_id).get_text().translate(translator) 
               for this_id in paragraph_ids]
    all_content += [content]
    sys.stdout.flush()
sys.stdout.write("\rWebscrape Progress: 100%\n")
sys.stdout.write("Webscrape complete!\n")

quotes = pd.DataFrame(columns = list(talks.columns))

for i in range(np.shape(talks)[0]):
    perc_comp = i / np.shape(talks)[0] * 100
    sys.stdout.write(f"\rData Compiling Progress: {int(np.round(perc_comp))}%")
    num_quotes = len(all_content[i])
    talk_speaker = [talks.iloc[i,0] for x in range(num_quotes)]
    talk_date = [talks.iloc[i,1] for x in range(num_quotes)]
    talk_title = [talks.iloc[i,2] for x in range(num_quotes)]
    talk_URL = [talks.iloc[i,3] for x in range(num_quotes)]
    
    talk_quote_info = pd.DataFrame({'Speaker':talk_speaker,
                                     'Date':talk_date,
                                     'Title':talk_title,
                                     'URL':talk_URL,
                                     'Content':all_content[i]})
    quotes = quotes.append(talk_quote_info, ignore_index = True)
sys.stdout.write("\rData Compiling Progress: 100%\n")
sys.stdout.write("Compilation complete!\n")

quotes.to_csv('Talk_Quotes_Data.csv', index = False)


##############################################################################
#           Scraping BYU Speeches

speeches_URL = 'https://speeches.byu.edu'

speakers = list(pd.read_csv('Talk_Quotes_Data.csv')['Speaker'].unique())
speakers_format = [spkr.lower().replace(' ','-').replace('.','') 
                   for spkr in speakers]

all_speaker_links = [speeches_URL + '/speakers/' + spkr + '/' 
                     for spkr in speakers_format]

speeches = pd.DataFrame(columns = ['Speaker', 'Date', 'Title', 'URL'])

escapes = ''.join([chr(char) for char in range(1, 32)])
translator = str.maketrans('', '', escapes)

for spkr_link in all_speaker_links:
    spkr_page = requests.get(spkr_link)
    spkr_soup = BeautifulSoup(spkr_page.content, 'lxml')
    all_talks = spkr_soup.find_all('article', 'card card--reduced')
    this_speaker = spkr_soup.find('h1', 'speaker-listing__name').text
    for talk in all_talks:
        this_Date = datetime.strptime(talk
                    .find('div', 'card__bylines card__bylines--reduced')
                    .text
                    .translate(translator), "%B %d, %Y")
        this_URL = talk.find('a')['href']
        this_Title = talk.find('h2').text.translate(translator)
        this_speech = pd.DataFrame({'Speaker':[this_speaker],
                                    'Date':[this_Date],
                                    'Title':[this_Title],
                                    'URL':[this_URL]})
        speeches = speeches.append(this_speech, ignore_index = True)
      

all_content = []
unavail_text = 'The text for this speech is unavailable.'
int_prop = 'Intellectual Reserve'
rights = 'All rights reserved.'

rows_to_rm = []

for speech_row in range(np.shape(speeches)[0]):
    perc_comp = speech_row / np.shape(speeches)[0] * 100
    sys.stdout.write(f"\rWebscrape Progress: {int(np.round(perc_comp))}%")
    speech_URL = speeches['URL'][speech_row]
    speech_page = requests.get(speech_URL)
    speech_soup = BeautifulSoup(speech_page.content, 'lxml')
    all_paragraphs = speech_soup.findChildren('p', 
                                              recursive = True, 
                                              attrs={'class': None})
    all_text = [p.text.translate(translator) for p in all_paragraphs]
    if len(all_text) > 2:
        end_idx = np.max(np.where([' amen.' in text 
                                   or ' Amen.' in text 
                                   or int_prop in text
                                   or unavail_text in text
                                   or rights in text
                                   for text in all_text]))
        if (unavail_text in all_text[end_idx] or 
            int_prop in all_text[end_idx] or 
            rights in all_text[end_idx]):
            end_idx += -1
            all_text = [text 
                        for text in all_text 
                        if 'Speech highlights' not in text]
        content = all_text[:(end_idx+1)]
        all_content += [content]
    else:
        rows_to_rm += [speech_row]
    sys.stdout.flush()

speeches = speeches[~speeches.index.isin(rows_to_rm)]

sys.stdout.write("\rWebscrape Progress: 100%\n")
sys.stdout.write("Webscrape complete!\n")

quotes = pd.DataFrame(columns = list(speeches.columns))

for i in range(np.shape(speeches)[0]):
    perc_comp = i / np.shape(speeches)[0] * 100
    sys.stdout.write(f"\rData Compiling Progress: {int(np.round(perc_comp))}%")
    num_quotes = len(all_content[i])
    speech_speaker = [speeches.iloc[i,0] for x in range(num_quotes)]
    speech_date = [speeches.iloc[i,1] for x in range(num_quotes)]
    speech_title = [speeches.iloc[i,2] for x in range(num_quotes)]
    speech_URL = [speeches.iloc[i,3] for x in range(num_quotes)]
    
    speech_quote_info = pd.DataFrame({'Speaker':speech_speaker,
                                     'Date':speech_date,
                                     'Title':speech_title,
                                     'URL':speech_URL,
                                     'Content':all_content[i]})
    quotes = quotes.append(speech_quote_info, ignore_index = True)
sys.stdout.write("\rData Compiling Progress: 100%\n")
sys.stdout.write("Compilation complete!\n")

quotes.to_csv('Speech_Quotes_Data.csv', index = False)



##############################################################################
#           Feature Selection

talks = pd.read_csv('Talk_Quotes_Data.csv', parse_dates = [1])
speeches = pd.read_csv('Speech_Quotes_Data.csv', parse_dates = [1])

all_talks = talks.append(speeches, ignore_index=True)
all_talks = all_talks[all_talks.Content.str.len() > 30]
np.shape(all_talks)

del talks, speeches

X_train_all, X_test_all, y_train, y_test = train_test_split(all_talks, 
                                                            all_talks.Speaker, 
                                                            test_size=0.2)
X_train = X_train_all.Content
X_test = X_test_all.Content
speaker_dict = {k:v for v, k in enumerate(all_talks.Speaker.unique())}
y_train = [speaker_dict[speaker] for speaker in list(y_train)]
y_test = [speaker_dict[speaker] for speaker in list(y_test)]
to_speaker_dict = {v:k for v, k in enumerate(all_talks.Speaker.unique())}


vectorizer = TfidfVectorizer(analyzer = 'word', 
                             stop_words = 'english',
                             max_features = 2500,
                             ngram_range = (1,5))
vectorizer.fit(list(all_talks.Content))
tfidf_X_train = vectorizer.transform(X_train)
tfidf_X_test = vectorizer.transform(X_test)


scaler = StandardScaler(with_mean=False)
tfidf_X_train = scaler.fit_transform(tfidf_X_train)
tfidf_X_test = scaler.transform(tfidf_X_test)
feature_names = vectorizer.get_feature_names()



##############################################################################
#           Modeling Speaker Classification


### MLR
# Grid search for MLR regularization
MLR_param_grid = {'C' : np.logspace(-4, 1, 6)}

# MLR model details
MLR_mod = LogisticRegression(random_state=0, 
                             multi_class = 'multinomial', 
                             penalty = 'l1',
                             solver = 'saga')

# MLR fitting
MLR_cv=GridSearchCV(MLR_mod, MLR_param_grid, cv=4 , verbose = 5, n_jobs=-1)
MLR_cv.fit(tfidf_X_train, y_train)

### SVM
# Grid search for SVM regularization and kernel function
SVM_param_grid = {'C' : np.logspace(-4, 2, 4),
              'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}

# SVM model details
SVM_model = SVC(verbose = True)

# SVM fitting
SVM_cv = GridSearchCV(SVM_model,
                      SVM_param_grid,
                      cv = 2, verbose = 8, n_jobs = -1)
SVM_cv.fit(tfidf_X_train, y_train)

##############################################################################
#           Prediction Measurements for Speaker Classification

### MLR
# MLR Best Regularization: lambda = 1/C 
1/MLR_cv.best_params_['C']
# MLR Run time
MLR_cv.refit_time_

# MLR Prediction
out_predictions_MLR = MLR_cv.predict(tfidf_X_test)

# MLR Speaker-Specific Recall
MLR_Results = pd.DataFrame({'Preds':out_predictions_MLR, 'Act':y_test})
MLR_Results['Accuracy'] = MLR_Results.Preds == MLR_Results.Act
MLR_Results.groupby('Act')['Accuracy'].mean()

# MLR F-score
f1_score(y_test, out_predictions_MLR, average = "weighted")

# MLR Confusion Matrix
pd.DataFrame(confusion_matrix(y_test, out_predictions_MLR), 
             index = speakers, columns = speakers)

### SVM
# SVM Best Regularization and Kernel
SVM_cv.best_params_
# SVM Run time
SVM_cv.refit_time_

# SVM Prediction
out_predictions_SVM = SVM_cv.predict(tfidf_X_test)

# SVM Speaker-Specific Recall
SVM_Results = pd.DataFrame({'Preds':out_predictions_SVM, 'Act':y_test})
SVM_Results['Accuracy'] = SVM_Results.Preds == SVM_Results.Act
SVM_Results.groupby('Act')['Accuracy'].mean()

# SVM F-score
f1_score(y_test, out_predictions_SVM, average = "weighted")

# SVM Confusion Matrix
pd.DataFrame(confusion_matrix(y_test, out_predictions_SVM), 
             index = speakers, columns = speakers)


##############################################################################
#           Modeling Dallin H. Oaks 2020

# Data prep for Dallin H. Oaks
tfidf_X_train_oaks = tfidf_X_train[X_train_all.Speaker=="Dallin H. Oaks"]
tfidf_X_test_oaks = tfidf_X_test[X_test_all.Speaker=="Dallin H. Oaks"]

y_train_oaks_full = pd.concat([X_train_all.Speaker=="Dallin H. Oaks",
                          X_train_all.Date.astype(str).str.startswith("2020")], 
                         axis = 1)
y_train_oaks = y_train_oaks_full[y_train_oaks_full.Speaker].Date
y_test_oaks_full = pd.concat([X_test_all.Speaker=="Dallin H. Oaks",
                         X_test_all.Date.astype(str).str.startswith("2020")], 
                         axis = 1)
y_test_oaks = y_test_oaks_full[y_test_oaks_full.Speaker].Date


### MLR
# Grid search for MLR regularization
MLR_param_grid_oaks = {'C' : np.logspace(-10, 10, 20)}

# MLR model details
MLR_mod_oaks = LogisticRegression(random_state=0, 
                                  penalty = 'l1',
                                  solver = 'saga')

# MLR fitting
MLR_cv_oaks = GridSearchCV(MLR_mod, 
                           MLR_param_grid_oaks, 
                           cv=4 , verbose = 5, n_jobs=-1)
MLR_cv_oaks.fit(tfidf_X_train_oaks, y_train_oaks)

### SVM
# Grid search for SVM regularization and kernel function
SVM_param_grid_oaks = {'C' : np.logspace(-4, 4, 8),
              'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}

# SVM model details
SVM_model_oaks = SVC(verbose = True)

# SVM fitting
SVM_cv_oaks = GridSearchCV(SVM_model_oaks,
                      SVM_param_grid_oaks,
                      cv=2, verbose = 8, n_jobs = -1)
SVM_cv_oaks.fit(tfidf_X_train_oaks, y_train_oaks)

##############################################################################
#           Prediction Measurements for Dallin H. Oaks 2020

### MLR
# MLR Best Regularization: lambda = 1/C 
1/MLR_cv_oaks.best_params_['C']
# MLR Run time
MLR_cv_oaks.refit_time_

# MLR Prediction
out_predictions_MLR_oaks = MLR_cv_oaks.predict(tfidf_X_test_oaks)
out_predictions_MLR_oaks_probs = MLR_cv_oaks.predict_proba(tfidf_X_test_oaks)

# MLR Speaker-Specific Recall
MLR_Results_oaks = pd.DataFrame({'Preds':out_predictions_MLR_oaks*1, 
                                 'Act':y_test_oaks*1})
MLR_Results_oaks['Accuracy'] = MLR_Results_oaks.Preds == MLR_Results_oaks.Act
MLR_Results_oaks.groupby('Act')['Accuracy'].mean()

# MLR F-score
f1_score(y_test_oaks, out_predictions_MLR_oaks, average = "weighted")

# MLR Confusion Matrix
pd.DataFrame(confusion_matrix(y_test_oaks, out_predictions_MLR_oaks), 
             index = ['Before', 'After'], columns = ['Before', 'After'])

# MLR ROC and AUC
MLR_fpr, MLR_tpr, MLR_thresholds = roc_curve(y_test_oaks*1, 
                                             out_predictions_MLR_oaks_probs[:,1])
auc(MLR_fpr, MLR_tpr)


### SVM
# SVM Best Regularization and Kernel
SVM_cv_oaks.best_params_
# SVM Run time
SVM_cv_oaks.refit_time_

# SVM Prediction
out_predictions_SVM_oaks = SVM_cv_oaks.predict(tfidf_X_test_oaks)
out_predictions_SVM_oaks_probs = SVM_cv_oaks.predict_proba(tfidf_X_test_oaks)

# SVM Speaker-Specific Recall
SVM_Results_oaks = pd.DataFrame({'Preds':out_predictions_SVM_oaks*1,
                                 'Act':y_test_oaks*1})
SVM_Results_oaks['Accuracy'] = SVM_Results_oaks.Preds == SVM_Results_oaks.Act
SVM_Results_oaks.groupby('Act')['Accuracy'].mean()

# SVM F-score
f1_score(y_test_oaks, out_predictions_SVM_oaks, average = "weighted")

# SVM Confusion Matrix
pd.DataFrame(confusion_matrix(y_test_oaks, out_predictions_SVM_oaks), 
             index = ['Before', 'After'], columns = ['Before', 'After'])

# SVM ROC and AUC
SVM_fpr, SVM_tpr, SVM_thresholds = roc_curve(y_test_oaks*1, 
                                             out_predictions_SVM_oaks_probs[:,1])
auc(SVM_fpr, SVM_tpr)


##############################################################################
#           Feature Selection for Talk-level

all_talks.Content += " "
talk_level = (all_talks
                  .groupby(['Speaker','Date'])['Content']
                  .sum()
                  .reset_index()
                  .Content)

(X_train_all_talks, 
     X_test_all_talks, 
     y_train_talks, 
     y_test_talks) = train_test_split(all_talks, 
                                      all_talks.Speaker,
                                      test_size=0.2)
X_train_talks = X_train_all_talks.Content
X_test_talks = X_test_all_talks.Content
y_train_talks = [speaker_dict[speaker] for speaker in list(y_train_talks)]
y_test_talks = [speaker_dict[speaker] for speaker in list(y_test_talks)]
to_speaker_dict = {v:k for v, k in enumerate(all_talks.Speaker.unique())}

tfidf_X_train_talks = vectorizer.transform(X_train_talks)
tfidf_X_test_talks = vectorizer.transform(X_test_talks)

tfidf_X_train_talks = scaler.fit_transform(tfidf_X_train_talks)
tfidf_X_test_talks = scaler.transform(tfidf_X_test_talks)

##############################################################################
#           Modeling Talk-Level

### MLR
# Grid search for MLR regularization
MLR_param_grid_talks = {'C' : np.logspace(-10, 10, 5)}

# MLR model details
MLR_mod_talks = LogisticRegression(random_state=0, 
                                  penalty = 'l1',
                                  solver = 'saga')

# MLR fitting
MLR_cv_talks = GridSearchCV(MLR_mod_talks, 
                           MLR_param_grid_talks, 
                           cv=4 , verbose = 5, n_jobs=-1)
MLR_cv_talks.fit(tfidf_X_train_talks, y_train_talks)

### SVM
# Grid search for SVM regularization and kernel function
SVM_param_grid_talks = {'C' : np.logspace(-4, 2, 2),
              'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}

# SVM model details
SVM_model_talks = SVC(verbose = True)

# SVM fitting
SVM_cv_talks = GridSearchCV(SVM_model_talks,
                      SVM_param_grid_talks,
                      cv=2, verbose = 10, n_jobs = -1)
SVM_cv_talks.fit(tfidf_X_train_talks, y_train_talks)

##############################################################################
#           Prediction Measurements for Talk-Level

### MLR
# MLR Best Regularization: lambda = 1/C 
1/MLR_cv_talks.best_params_['C']
# MLR Run time
MLR_cv_talks.refit_time_

# MLR Prediction
out_predictions_MLR_talks = MLR_cv_talks.predict(tfidf_X_test_talks)

# MLR Speaker-Specific Recall
MLR_Results_talks = pd.DataFrame({'Preds':out_predictions_MLR_talks, 
                                  'Act':y_test_talks})
MLR_Results_talks['Accuracy'] = MLR_Results_talks.Preds==MLR_Results_talks.Act
MLR_Results_talks.groupby('Act')['Accuracy'].mean()

# MLR F-score
f1_score(y_test_talks, out_predictions_MLR_talks, average = "weighted")

# MLR Confusion Matrix
pd.DataFrame(confusion_matrix(y_test_talks, out_predictions_MLR_talks), 
             index = speakers, columns = speakers)

### SVM
# SVM Best Regularization and Kernel
SVM_cv_talks.best_params_
# SVM Run time
SVM_cv_talks.refit_time_

# SVM Prediction
out_predictions_SVM_talks = SVM_cv_talks.predict(tfidf_X_test_talks)

# SVM Speaker-Specific Recall
SVM_Results_talks = pd.DataFrame({'Preds':out_predictions_SVM_talks,
                                  'Act':y_test_talks})
SVM_Results_talks['Accuracy'] = SVM_Results_talks.Preds==SVM_Results_talks.Act
SVM_Results_talks.groupby('Act')['Accuracy'].mean()

# SVM F-score
f1_score(y_test_talks, out_predictions_SVM_talks, average = "weighted")

# SVM Confusion Matrix
pd.DataFrame(confusion_matrix(y_test_talks, out_predictions_SVM_talks), 
             index = speakers, columns = speakers)

