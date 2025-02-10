# Helper functions for data processing our corpora
# Save this script in the same directory as other notebooks and import as "import engCDSutils as util"

import os
import pandas as pd
import ast
import subprocess
from collections import Counter
import math
import numpy as np
import matplotlib.pyplot as plt

def preProcessCSV(df,properNames):
    '''
    Returns corpus text after removing CHAT codes, punctuation, names of people
    Input: df of full corpus scrape from CHILDES
    Output: df of full corpus scrape from CHILDES but with a cleaner gloss column
    '''

    #words = text.split() 
    #words = text
   
    phrase_list = df['gloss']
    cleaned_phrases = []

    thrown_out = []
    
    for phrase in phrase_list:
        words = phrase.split()
        cleaned_words = []
        for word in words:
            if word in ["xxx" , "yyy","www"]:
                thrown_out.append(word)
                word = ""

            if len(word)>2:
                if word[-2:] == "'s":
                    word = word[:-2]+"s"


            cleaned_word = (word.strip(".,!?;'':\"()[]{}")).lower()

            if "_" in cleaned_word:
                cleaned_word = cleaned_word.replace("_", " ")

            if "+" in cleaned_word:
                cleaned_word=cleaned_word.replace("+", " ")

            if "-" in cleaned_word:
                cleaned_word=cleaned_word.replace("-", " ")

            if " " in cleaned_word:    
                cleaned_words = cleaned_words + cleaned_word.split()
                continue

            if cleaned_word in properNames:
                thrown_out.append(word)
                cleaned_word = ""


            cleaned_words.append(cleaned_word.strip())
            
        cleaned_phrase = ' '.join(cleaned_words)
        cleaned_phrases.append(cleaned_phrase)
    
 
    df['gloss_cleaned']  = cleaned_phrases
     
    return df
    

def preProcess(text, properNames):
    '''
    Returns corpus text after removing CHAT codes, punctuation, names of people
    Input : List of all utterances in corpus, list of proper nouns
    Output: list of cleaned text, no. of tokens after cleaning, no. of tokens thrown out
    '''

    #words = text.split() 
    #words = text
    words = []
    for phrase in text:
        words+=phrase.split()
    
    
    print("Total number of word tokens originally in corpus:", len(words))
    
    
    cleaned_words = []
    thrown_out = []
    for word in words:
        if word in ["xxx" , "yyy","www"]:
            thrown_out.append(word)
            word = ""
        
        if len(word)>2:
            if word[-2:] == "'s":
                word = word[:-2]+"s"
                
            
        cleaned_word = (word.strip(".,!?;'':\"()[]{}")).lower()
        
        if "_" in cleaned_word:
            cleaned_word = cleaned_word.replace("_", " ")

        if "+" in cleaned_word:
            cleaned_word=cleaned_word.replace("+", " ")
        
        if "-" in cleaned_word:
            cleaned_word=cleaned_word.replace("-", " ")
            
        if " " in cleaned_word:    
            cleaned_words = cleaned_words + cleaned_word.split()
            continue
            
        if cleaned_word in properNames:
            thrown_out.append(word)
            cleaned_word = ""
            
        
        cleaned_words.append(cleaned_word.strip())
        
    
    print("Total number of word tokens found during preprocessing:", len(cleaned_words))
    
    nonempty = [w for w in cleaned_words if w!=""]
    
    print("Total number of word tokens left after preprocessing:", len(nonempty))
    print("Total number of UNIQUE word types left after preprocessing:", len(set(nonempty)))
    print("Total number of word tokens we threw out:", len(thrown_out))    
    return cleaned_words, len(cleaned_words), thrown_out


def giveDictionary(linktodict):
    
    '''
    Returns a dictionary of common words and their IPA (first pass of g2p)
    Input: dictionary to refer to
    Output: words and their transcriptions in dict form
    '''
    
    dict_file_path = linktodict
    word_dict = {}
    
    with open(dict_file_path, 'r') as file:
        for line in file:
            if line.strip():
                 
                parts = line.strip().split()
                word = parts[0]
                notword = parts[1:]
                transcription = []
                for ch in notword:
                    try:
                        float(ch)
                    except ValueError:
                        transcription.append(ch) 
                phonemes = transcription 
                word_dict[word] = phonemes
    return word_dict


def createTranscription(cleaned_text,link_to_dict):
    '''
    Returns a dictionary of corpus words and their IPA, if available. Otherwise, entry value is empty string.
    Input: list of cleaned words to transcribe, dictionary to use
    Output: dictionary of transcriptions, length of cleaned word list, unrecognized tokens
    '''
    
    #with open(linktotext, "r") as file:
    #    text = file.read()

    transcribed_text = {}
    unrecog_ctr =0
    transcription_dict = giveDictionary(link_to_dict)
    
    for word in cleaned_text:
        cleaned_word = word.strip(".,!?;'':\"()[]{}")
        transcribed_word = transcription_dict.get(cleaned_word.lower(), "")
        transcribed_text[word] = transcribed_word
        if transcribed_word == '':
            unrecog_ctr+=1
        
    
    return transcribed_text, len(cleaned_text),unrecog_ctr


def prepareGlossedCSV(df, corpus_name,dictionary,directory_of_output):
    '''
    Returns the dictionary it used and a dataframe of each utterance glossed in IPA. Also saves it as a csv.
    Output: dictionary used to create gloss, glossed df, words which are still not found
    '''
    #df_path = os.path.join(directory_of_csv,name+".csv")
    #df = pd.read_csv(df_path)
    result = []
    stillerror = []
     
    
    for utterance in df['gloss_cleaned']:
        utt_result = []
        utterance = str(utterance)
        splitutt = utterance.split()

        for word in splitutt:
            try:
                utt_result.append(dictionary[word])
                if dictionary[word] =='':
                    stillerror.append(word)
            except KeyError:
                stillerror.append(word)
                utt_result.append("")
                
        result.append(utt_result)


    df['phonemic_gloss'] = result
    csvpath = os.path.join(directory_of_output, corpus_name+'_ipa.csv')
    df.to_csv(csvpath, index=False)
    return dictionary, df,stillerror




# shell script template
template = """
#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=2:00:00,h_data=1G
#$ -pe shared 8
#$ -m a


# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo "Running on: " {input_file}
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
source $HOME/.bash_profile
cd LINK_TO_PROJECT_DIRECTORY


module load python/3.9.6
module load anaconda3
conda activate aligner
module load gcc

mfa g2p -n 1 --no_overwrite --use_mp {input_file} {g2p_model} {output_file}

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
#### submit_job.sh STOP ####
"""


def create_and_submit_job(input_file, g2p_model, output_file):
     
    script_content = template.format(g2p_model=g2p_model, output_file=output_file,input_file=input_file )
     
     
    script_filename = 'job_script_'+'item_name'+ '.sh'
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    
    result = subprocess.run(["qsub", script_filename], capture_output=True, text=True)
    
     
    print(result.stdout) 
    print(result.stderr)


def tokenDescriptions(directory_of_output,corpus_name):
    ipa_output_path = os.path.join(directory_of_output, corpus_name+'_ipa_result.csv')
    result_df = pd.read_csv(ipa_output_path)   
    how_many_words = []
    how_many_phonemes = []
    how_many_transcribed = []
    for i in result_df['phonemic_gloss']:
        x = ast.literal_eval(i)
        how_many_words.append(len(x))
        ctr = 0
        phonemecount = 0
        for y in x: 
            phonemecount+=len(y)
            if y!='':
                ctr +=1
        how_many_transcribed.append(ctr)
        how_many_phonemes.append(phonemecount)

    result_df['length_of_result'] = how_many_words
    result_df['num_relevant_words'] = how_many_transcribed
    result_df['num_phonemes'] = how_many_phonemes
    print("Total number of words we had from CHILDES originally", result_df['num_tokens'].sum())
    print("Total number of words we had after initial cleaning", result_df['length_of_result'].sum())
    print("Total number of words we have transcriptions for", result_df['num_relevant_words'].sum())
    print("Total number of phonemes we have ", result_df['num_phonemes'].sum())
    print("Total number of  unique child IDs detected:",len(set(list(result_df['target_child_id']))))
    print("Total number of  unique speaker roles detected:",len(set(list(result_df['speaker_role']))))
    
    subset_df = result_df[['gloss','target_child_name' ,'transcript_id' ,'target_child_age','target_child_sex','target_child_id','phonemic_gloss','num_relevant_words','num_phonemes']]
    aggregated_df = subset_df.groupby(['target_child_age', 'target_child_sex']).agg({'num_relevant_words': 'sum','num_phonemes': 'sum'})
    result_df.to_csv(ipa_output_path,index=False)
    return subset_df, aggregated_df
    

def ageSexSummary(df,sex=True):

    if sex == False:
        a = df.groupby(['age_bin']).agg(total_words = ('num_relevant_words', 'sum'),total_phonemes = ('num_phonemes','sum'), no_unique_child = ('child_id', 'nunique')).reset_index()

    elif 'age_bin' in df.columns:
        a = df.groupby(['age_bin', 'target_child_sex']).agg(total_words = ('num_relevant_words', 'sum'),total_phonemes = ('num_phonemes','sum'), no_unique_child = ('child_id', 'nunique')).reset_index()
    else:
        a = df.groupby(['target_child_age', 'target_child_sex']).agg(total_words = ('num_relevant_words', 'sum'),total_phonemes = ('num_phonemes','sum'), no_unique_child = ('child_id', 'nunique')).reset_index()
    return a

def binInto6(subset):
    
    subset['age_bin'] =  pd.cut(subset['target_child_age'], [i for i in range(3,52,6)], right=False, labels=[str(i)+"-"+str(i+6) for i in range(3,47,6)],  include_lowest=False,   ordered=True)
    return subset


def samplePhonemes(N, scrape, age, seed=0):
    '''
    Input:  N = number of phonemes to be sampled
            scrape = full corpora text available to use
            age = age bin to sample from
    
    Returns a dataframe with randomized utterances such that utterances are kept in-tact and the sum of phonemes is as close to N as possible
    
    '''
    
    df = scrape[scrape['age_bin'] == age]
    df_shuffled = df.sample(frac = 1,random_state=seed) #shuffle data
    df_shuffled['cumulative'] = df_shuffled.num_phonemes.cumsum() 
    return df_shuffled[df_shuffled.num_phonemes.cumsum() <= N] #as close to N as posssible

def collectAllWords(df):
    '''
    Makes a list of all the words that we have in our sample
    '''
    word_list = []
    for utterance in df['phonemic_gloss_updated']:
        word_list += utterance
    return word_list

def collectAllPhonemes(wordlist):
    phon_list=[]
    for word in wordlist:
        phon_list += word
    return phon_list



def getRelFreq(sample):
    wordlist = collectAllWords(sample)
    res = Counter(collectAllPhonemes(wordlist))
    relative_dict = {}
    total_phonemes = sample.num_phonemes.sum()
    for key,value in res.items():
        relative_dict[key] = value/total_phonemes
    return relative_dict

def getLogFreq(sample):
    '''
        Copmputes -log2(relative frequency) of each phoneme in the sample
    '''
    
    wordlist = collectAllWords(sample)
    res = Counter(collectAllPhonemes(wordlist))
    relative_dict = {}
    total_phonemes = sample.num_phonemes.sum()
    for key,value in res.items():
        relative_dict[key] = -1*math.log2(value/total_phonemes)
    
    average_value = np.mean(list(relative_dict.values()))
    return relative_dict

def getContexts(sample):
    '''
        List all possible contexts of each phoneme in the sample
        Returns a dictionary of the form: key = phoneme, value = list of lists i.e. a list of contexts with repetition
    '''
    
    wordlist  = collectAllWords(sample)
    phonlist = collectAllPhonemes(wordlist)
     
    phoneme_contexts = {}
    
    for i in phonlist:
        phoneme_contexts[i] = []
        
    for word in wordlist:
        for phoneme in word:
            phoneme_contexts[phoneme].append( word[:word.index(phoneme)])
             
    return phoneme_contexts 

def getPredic(phoneme, given_context, contexts_from_sample):
    '''
      Returns -log(P(phoneme | context)) where context is preceding phonemes in a word
    
    '''
    relevant_contexts = contexts_from_sample[phoneme]
    numerator = 0
     
    for context in relevant_contexts:
            if context == given_context:
                numerator+=1
    
    denominator = 0
    all_contexts = list(contexts_from_sample.values())
    
    for contextlist in all_contexts:
        for context in contextlist:
            if context == given_context:
                denominator+=1    
            
    return -1 * np.log2(numerator/denominator)

def getInfor(phoneme,contexts_from_sample):
    
    '''
       Returns informativity (weighted average of predictability) of a phoneme
    
    '''
    
    relevant_contexts = contexts_from_sample[phoneme]
    phoneme_occur = len(relevant_contexts)
     
    info = 0
    contexts_unique = []
    
    for context in relevant_contexts:
        if not context in contexts_unique:
            contexts_unique.append(context)
    
    for context in contexts_unique:
        predict = getPredic(phoneme, context,contexts_from_sample)
        ctr = 0
        for i in relevant_contexts:
            if context == i:
                ctr+=1
        conditional = ctr/phoneme_occur
       
        info += (conditional * predict)
    
    return info

def getSuccessiveSamples(link_to_samplingResults):

    '''
        Creates a series of samples of increasing size and saves the successive samples per age bin
    '''
    
    N_sizes = [i for i in range(3000,100000,3000)]
    agebins = [str(i)+"-"+str(i+6) for i in range(3,47,6)]
    mean_logfreq = []
    mean_infor = []
    df_N = []
    df_age = []
    
    
    for agebin in agebins:
            for N in N_sizes:
                sample = samplePhonemes(N,  giant_scrape, agebin)
                contexts_from_sample = getContexts(sample) 
                list_of_phonemes_in_sample = list(contexts_from_sample.keys())
                freq_results = getLogFreq(sample)
 
                freq_values = []
                info_values = []

                for phoneme in list_of_phonemes_in_sample:
                    freq_values.append(freq_results[phoneme])
                    info_values.append(getInfor(phoneme,contexts_from_sample)) 
                    
                    
                name_of_file = "sample"+str(N)+"age"+agebin
                df_result = pd.DataFrame({'phoneme': list_of_phonemes_in_sample, 'logfreq':freq_values,  'informativity':info_values})
                df_result.to_csv(os.path.join(link_to_samplingResults, name_of_file+'.csv'))
                mean_logfreq.append(df_result['logfreq'].mean())
                mean_infor.append(df_result['informativity'].mean())
                df_N.append(N)
                df_age.append(agebin)
                
    mean_result = pd.DataFrame({'sample_size': df_N, 'age_bin':df_age,  'mean_informativity':mean_infor, 'mean_logfreq':mean_logfreq})
    mean_result.to_csv(os.path.join(link_to_samplingResults,  'mean_results.csv'))



def createSampleSizePlot(windowsize:int, link_to_mean_csv):

    '''
        Creates a plot of rolling standard deviation of informativity and log frequency for each age bin
    '''
    agebins = [str(i)+"-"+str(i+6) for i in range(3,47,6)]
    overall_df = pd.read_csv(link_to_mean_csv)
    for age in agebins:
        overall_df_subset = overall_df[overall_df['age_bin'] == age]
        overall_df_subset['rolling_std_infor'] = overall_df_subset.mean_informativity.rolling(windowsize).std()
        overall_df_subset['rolling_std_logfreq'] = overall_df_subset.mean_logfreq.rolling(windowsize).std()
    
        plt.figure(figsize=(10, 6))
        plt.plot(overall_df_subset['sample_size'], overall_df_subset['rolling_std_logfreq'], color='green', label='Average phonemic log freq')
        plt.plot(overall_df_subset['sample_size'], overall_df_subset['rolling_std_infor'], color='blue', label='Average phonemic informativity')

        # Add titles and labels
        plt.title('Rolling Standard Deviations: AGE GROUP '+age)
        plt.xlabel('Sample Size (N = no. of phonemes)')
        plt.ylabel('Rolling Standard Deviation, window size '+str(windowsize))
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()
        
        print('The first time the rolling std for logfreq falls below 0.01 is' ,overall_df_subset[overall_df_subset['rolling_std_logfreq']<0.01]['sample_size'].head(1))
        print('The first time the rolling std for informativity falls below 0.01 is' ,overall_df_subset[overall_df_subset['rolling_std_infor']<0.01]['sample_size'].head(1))



def linePlots(metric,logfreq_melted,info_melted):
    '''
        Creates line plots of informativity and log frequency for each phoneme over age
    '''
    plt.figure(figsize=(6, 13))
    if metric == 'logfreq':
        df = logfreq_melted
    else:
        df = info_melted
    
    for phoneme in df['phoneme'].cat.categories:
        df_phoneme = df[df['phoneme'] == phoneme]
        plt.plot(df_phoneme['age'], df_phoneme['value'], marker='o', label=phoneme)
   
    plt.xlabel('Age')
    plt.ylabel('Value')
    plt.title(f'Variable: {metric}')
    plt.legend(title='Phoneme',loc='upper right'  , bbox_to_anchor=(1.7,1),ncol=3)
    plt.grid(True)
    plt.show()