#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Miguel Angel Sanchez Perez'
__email__ = 'masp1988 at hotmail dot com'
__version__ = '2.0'

"""(1) Miguel A. Sanchez-Perez, Alexander F. Gelbukh, Grigori Sidorov: Adaptive Algorithm for Plagiarism Detection: The Best-Performing Approach at PAN 2014 Text Alignment Competition. CLEF 2015: 402-413"""

import os
import sys
import xml.dom.minidom
import codecs
import nltk
import Stemmer
import math
import time
import copy
import string
import xml.etree.ElementTree as ET

def sum_vect(dic1, dic2):
    """
    DESCRIPTION: Adding two vectors in form of dictionaries (sparse vectors or inverted list)  
    INPUTS: dic1 <dictionary> - Vector 1
            dic2 <dictionary> - Vector 2
    OUTPUT: res <dictionary> - Sum of the two vectors
    """
    res = copy.deepcopy(dic1)
    for i in dic2.keys():
        if res.has_key(i):
            res[i] += dic2[i]
        else:
            res[i] = dic2[i]
    return res

def ss_treat(list_dic, offsets, min_sentlen, rssent, voc):
    """
    DESCRIPTION: Remove or annex sentences with less than a certain amount of words (min_sentlen)
    INPUTS: list_dic <list of dictionaries> - List containing the vectors (dictionaries) of a document
            offsets <2-tuple> - First value contains the offset character and the next contain the length in characters
            min_sentlen <integer> - Minimum of words allowed in a sentence
            rssent <integer> - Action to perform (0): Annex small sentences, (1) Remove small sentences
            voc <dictionary> - The keys are the types (vocabulary) in a document while the values are the sentence frequency    
    OUTPUT: No returned value. Modify the inputs list_dic, offsets and voc
    """
    if rssent == 0: #Annexing small sentences
        i = 0
        range_i = len(list_dic) - 1
        while i < range_i:
            if sum(list_dic[i].values()) < min_sentlen:
                for k in list_dic[i].keys():
                    if list_dic[i + 1].has_key(k):
                        voc[k] -= 1
                list_dic[i + 1] = sum_vect(list_dic[i + 1], list_dic[i])
                del list_dic[i]
                offsets[i + 1] = (offsets[i][0], offsets[i + 1][1] + offsets[i][1])
                del offsets[i]
                range_i -= 1
            else:
                i = i + 1
    else: #Removing small sentences
        i = 0
        range_i = len(list_dic) - 1
        while i < range_i:
            if sum(list_dic[i].values()) < min_sentlen:
                del list_dic[i]
                del offsets[i]
                range_i -= 1
            else:
                i = i + 1

def tf_idf(list_dic1, voc1, list_dic2, voc2):
    """
    DESCRIPTION: Compute the tf-idf <tf x log(N/df)>  from a list of sentences with tf and the vocabularies in suspicios and source document   
    INPUT: list_dic1 <list of dictionaries> -  List containing the vectors (dictionaries) of document 1
           voc1 <dictionary> - Vocabulary at document 1 with the idf of each one
           list_dic2 <list of dictionaries> -  List containing the vectors (dictionaries) of document 2
           voc2 <dictionary> - Vocabulary at document 2 with the idf of each one
    OUTPUT: No returned value. Modify the inputs list_dic1 and list_dic2
    """
    df = sum_vect(voc1, voc2)
    td = len(list_dic1) + len(list_dic2)
    for i in range(len(list_dic1)):
        for j in list_dic1[i].keys():
            list_dic1[i][j] *= math.log(td / float(df[j]))        
    for i in range(len(list_dic2)):
        for j in list_dic2[i].keys():
            list_dic2[i][j] *= math.log(td / float(df[j]))

def tf_idf_hard(list_dic1, voc1, list_dic2, voc2):
    """
    DESCRIPTION: Compute the tf-idf <tf x [log(N/df)]^2>  from a list of sentences with tf and the vocabularies in suspicios and source document  
    INPUT: list_dic1 <list of dictionaries> -  List containing the vectors (dictionaries) of document 1
           voc1 <dictionary> - Vocabulary at document 1 with the idf of each one
           list_dic2 <list of dictionaries> -  List containing the vectors (dictionaries) of document 2
           voc2 <dictionary> - Vocabulary at document 2 with the idf of each one
    OUTPUT: No returned value. Modify the inputs list_dic1 and list_dic2
    """
    df = sum_vect(voc1, voc2)
    td = len(list_dic1) + len(list_dic2)
    for i in range(len(list_dic1)):
        for j in list_dic1[i].keys():
            list_dic1[i][j] *= math.pow(math.log(td / float(df[j])), 2)
    for i in range(len(list_dic2)):
        for j in list_dic2[i].keys():
            list_dic2[i][j] *= math.pow(math.log(td / float(df[j])), 2)

def tf_idf_soft(list_dic1, voc1, list_dic2, voc2):
    """
    DESCRIPTION: Compute the tf-idf <tf x [log(N/df)]^0.5>  from a list of sentences with tf and the vocabularies in suspicios and source document  
    INPUT: list_dic1 <list of dictionaries> -  List containing the vectors (dictionaries) of document 1
           voc1 <dictionary> - Vocabulary at document 1 with the idf of each one
           list_dic2 <list of dictionaries> -  List containing the vectors (dictionaries) of document 2
           voc2 <dictionary> - Vocabulary at document 2 with the idf of each one
    OUTPUT: No returned value. Modify the inputs list_dic1 and list_dic2
    """
    df = sum_vect(voc1, voc2)
    td = len(list_dic1) + len(list_dic2)
    for i in range(len(list_dic1)):
        for j in list_dic1[i].keys():
            list_dic1[i][j] *= math.pow(math.log(td / float(df[j])), 0.5)
    for i in range(len(list_dic2)):
        for j in list_dic2[i].keys():
            list_dic2[i][j] *= math.pow(math.log(td / float(df[j])), 0.5)
     
def tf_idf_ind(list_dic1, voc1, list_dic2, voc2):
    """
    DESCRIPTION: Compute the tf-idf <tf x log(N/df)>  from a list of sentences with tf and the vocabulary of each document separately   
    INPUT: list_dic1 <list of dictionaries> -  List containing the vectors (dictionaries) of document 1
           voc1 <dictionary> - Vocabulary at document 1 with the idf of each one
           list_dic2 <list of dictionaries> -  List containing the vectors (dictionaries) of document 2
           voc2 <dictionary> - Vocabulary at document 2 with the idf of each one
    OUTPUT: No returned value. Modify the inputs list_dic1 and list_dic2
    """
    td1 = len(list_dic1)
    td2 = len(list_dic2)
    for i in range(td1):
        for j in list_dic1[i].keys():
            den = math.log(td1 / float(voc1[j]))
            if den != 0:
                list_dic1[i][j] /= den
            else:
                list_dic1[i][j] = 0
    for i in range(td2):
        for j in list_dic2[i].keys():
            list_dic2[i][j] *= math.log(td2 / float(voc2[j]))
            
def eucl_norm(d1):
    """
    DESCRIPTION: Compute the Euclidean norm of a sparse vector  
    INPUT: d1 <dictionary> - sparse vector representation
    OUTPUT: Norm of the sparse vector d1
    """
    norm = 0.0
    for val in d1.values():
        norm += float(val * val)
    return math.sqrt(norm)

def cosine_measure(d1, d2):
    """
    DESCRIPTION: Compute the cosine measure (cosine of the angle between two vectors) in sparse (dictionary) representation
    INPUT: d1 <dictionary> - Sparse vector 1
           d2 <dictionary> - Sparse vector 2
    OUTPUT: Cosine measure
    """
    dot_prod = 0.0
    det = eucl_norm(d1) * eucl_norm(d2)
    if det == 0:
        return 0 
    for word in d1.keys():
        if d2.has_key(word):
            dot_prod += d1[word] * d2[word]
    return dot_prod / det

def dice_coeff(d1, d2):
    """
    DESCRIPTION: Compute the dice coefficient in sparse (dictionary) representation  
    INPUT: d1 <dictionary> - Sparse vector 1
           d2 <dictionary> - Sparse vector 2
    OUTPUT: Dice coefficient
    """
    if len(d1) + len(d2) == 0:
        return 0
    intj = 0
    for i in d1.keys():
        if d2.has_key(i):
            intj += 1
    return 2 * intj / float(len(d1) + len(d2))

def adjacent_sents(a, b, th):
    """
    DESCRIPTION: Define if two sentences are adjacent measured in sentences
    INPUT: a <int> - Sentence a index,
           b <int> - Sentence b index
           th <int> - maximum gap between indexes
    OUTPUT: True if the two sentences are adjacents, False otherwise
    """
    if abs(a - b) - 1 <= th:
        return True
    else:
        return False

def adjacent_chars(a, b, offsets, th):
    """
    DESCRIPTION: Define if two sentences are adjacent measured in characters
    INPUT: a <int> - Sentence a index,
           b <int> - Sentence b index
           offsets <list of tuples (int, int)> - Contain the char offset and length of each sentence 
           th <int> - maximum gap between indexes
    OUTPUT: True if the two sentences are adjacents, False otherwise
    """
    if a > b:
        if offsets[a][0] + offsets[a][1] - offsets[b][0] - 1 <= th:
            return True
        else:
            return False
    else:
        if offsets[b][0] + offsets[b][1] - offsets[a][0] - 1 <= th:
            return True
        else:
            return False

def frag_founder(ps, src_offsets, susp_offsets, src_gap, susp_gap, src_size, susp_size, side):
    """
    DESCRIPTION: Form clusters by grouping "adjacent" sentences in a given side (source o suspicious) 
    INPUT: ps <list of tuples (int, int)> - Seeds
           src_offsets <list of tuples (int, int)> - Contain the char offset and length of each source document sentence
           susp_offsets <list of tuples (int, int)> - Contain the char offset and length of each suspicious document sentence
           src_gap <int> - Max gap between sentences to be consider adjacent in the source document
           susp_gap <int> - Max gap between sentences to be consider adjacent in the suspicious document
           src_size <int> - Minimum amount of sentences in a plagiarism case in the side of source document
           susp_size <int> - Minimum amount of sentences in a plagiarism case in the side of suspicious document
           side <0 or 1> 0: Suspicious document side, 1: Source document side   
    OUTPUT: res <list of list of tuples (int, int)> - Contains the clusters 
    """
    if side == 0:
        max_gap = susp_gap
        min_size = susp_size
        offsets = susp_offsets
    else:
        max_gap = src_gap
        min_size = src_size
        offsets = src_offsets
    res = []
    ps.sort(key = lambda tup: tup[side])
    sub_set = []
    for pair in ps:
        if len(sub_set) == 0:
            sub_set.append(pair)
        else:
            if adjacent_sents(pair[side], sub_set[-1][side], max_gap):
            #if adjacent_chars(pair[side], sub_set[-1][side], offsets, max_gap):
                sub_set.append(pair)
            else:
                if len(sub_set) >= min_size:
                    res.append(sub_set)
                sub_set = [pair]
    if len(sub_set) >= min_size:
        res.append(sub_set)
    return res

def clustering(ps, src_offsets, susp_offsets, src_gap, susp_gap, src_size, susp_size, side, times):
    """
    DESCRIPTION: Generates the clusters of seeds  
    INPUT: ps <list of tuples (int, int)> - Seeds
           src_offsets <list of tuples (int, int)> - Contain the char offset and length of each source document sentence
           susp_offsets <list of tuples (int, int)> - Contain the char offset and length of each suspicious document sentence
           src_gap <int> - Max gap between sentences to be consider adjacent in the source document
           susp_gap <int> - Max gap between sentences to be consider adjacent in the suspicious document
           src_size <int> - Minimum amount of sentences in a plagiarism case in the side of source document
           susp_size <int> - Minimum amount of sentences in a plagiarism case in the side of suspicious document
           side <0 or 1> 0: Suspicious document side, 1: Source document side
           times <int> - Counts how many times clustering() have been called   
    OUTPUT: res <list of list of tuples (int, int)> - Contains the clusters
    """
    ps_sets = frag_founder(ps, src_offsets, susp_offsets, src_gap, susp_gap, src_size, susp_size, side)
    res = []
    if len(ps_sets) <= 1 and times > 0:
        return ps_sets
    else:
        times += 1
        for i in ps_sets:
            partial_res = clustering(i, src_offsets, susp_offsets, src_gap, susp_gap, src_size, susp_size, (side + 1) % 2, times)
            res.extend(partial_res)
    return res
    
def validation(plags, psr, src_offsets, susp_offsets, src_bow, susp_bow, src_gap, src_gap_least, susp_gap, susp_gap_least, src_size, susp_size, th3):
    """
    DESCRIPTION: Compute the similarity of the resulting plagiarism cases from extension. In case of being below certain threshold extension is applied again with max_gap - 1 
    INPUT: plags <list of list of two tuples [(int, int), (int, int)]> - Have the plagiarism cases represented by min and max sentence index in suspicious and source document respectively
           psr <list of list of tuples (int, int)> - Contains the clusters
           src_offsets <list of tuples (int, int)> - Contain the char offset and length of each source document sentence
           susp_offsets <list of tuples (int, int)> - Contain the char offset and length of each suspicious document sentence
           src_bow <list of dictionaries> - Bag of words representing each sentence vector of source document
           susp_bow <list of dictionaries> - Bag of words representing each sentence vector of suspicious document
           src_gap <int> - Max gap between sentences to be consider adjacent in the source document
           src_gap_least <int> - Smallest value the max gap between sentences considerd adjacent can gets in the source document 
           susp_gap <int> - Max gap between sentences to be consider adjacent in the suspicious document
           susp_gap_least <int> - Smallest value the max gap between sentences considerd adjacent can gets in the suspicious document
           src_size <int> - Minimum amount of sentences in a plagiarism case in the side of source document
           susp_size <int> - Minimum amount of sentences in a plagiarism case in the side of suspicious document
           th3 <float> - Threshold for the minimum cosine similarity between source and suspicios fragments in a plagiarism case  
    OUTPUT: res_plags <list of list of two tuples [(int, int), (int, int)]> - Contains the plagiarism cases that passed the validation process
            res_psr <list of list of tuples (int, int)> - Contains the clusters that passed the validation process
            res_sim_frag <list of floats> - Stores the cosine similarity between source and suspicios fragments in the plagiarism cases
    """  
    res_plags = []
    res_psr = []
    res_sim_frag = []
    i = 0
    range_i = len(plags)
    while i < range_i:
        susp_d = {}
        for j in range(plags[i][0][0], plags[i][0][1] + 1):
            susp_d = sum_vect(susp_d, susp_bow[j])
        src_d = {}
        for j in range(plags[i][1][0], plags[i][1][1] + 1):
            src_d = sum_vect(src_d, src_bow[j])
        #if dice_coeff(src_d, susp_d) <= th3:# or cosine_measure(src_d, susp_d) <= 0.40:
        sim_frag = cosine_measure(src_d, susp_d) 
        if sim_frag <= th3:
            #print 'Did not passed with gap', src_gap, '!'
            if src_gap > src_gap_least and susp_gap > susp_gap_least:#Do until substraction +1
                new_psr = clustering(psr[i], src_offsets, susp_offsets, src_gap - 1, susp_gap - 1, src_size, susp_size, 0, 0)
                new_plags = []
                for ps_set in new_psr:
                    new_plags.append([(min([x[0] for x in ps_set]), max([x[0] for x in ps_set])), (min([x[1] for x in ps_set]), max([x[1] for x in ps_set]))])
                if len(new_plags) == 0:
                    return []
                temp_res = validation(new_plags, new_psr, src_offsets, susp_offsets,src_bow, susp_bow, src_gap - 1, src_gap_least, susp_gap - 1, susp_gap_least, src_size, susp_size, th3)###---
                if len(temp_res) == 0:
                    plags_rec, psr_rec, res_sim_frag_rec = [], [], []
                else:
                    plags_rec, psr_rec, res_sim_frag_rec = temp_res[0], temp_res[1], temp_res[2]
                if len(plags_rec) != 0:
                    res_plags.extend(plags_rec)
                    res_psr.extend(psr_rec)
                    res_sim_frag.extend(res_sim_frag_rec)
            #else:
                #print 'Not passed with the options allowed!'
            i += 1
        else:
            #print 'Passed with gap', src_gap,'!'
            res_plags.append(plags[i])
            res_psr.append(psr[i])
            res_sim_frag.append(sim_frag)
            i += 1
    return res_plags, res_psr, res_sim_frag

def remove_overlap3(plags, psr, src_bow, susp_bow):
    """
    DESCRIPTION: From a set of overlapping plagiarism cases, looking only on the suspicious side, selects the best case. See article (1) at the beggining of this file, for the formal description.  
    INPUT: plags <list of list of two tuples [(int, int), (int, int)]> - Have the plagiarism cases represented by min and max sentence index in suspicious and source document respectively
           psr <list of list of tuples (int, int)> - Contains the clusters
           src_bow <list of dictionaries> - Bag of words representing each sentence vector of source document
           susp_bow <list of dictionaries> - Bag of words representing each sentence vector of suspicious document
    OUTPUT: res_plags <list of list of two tuples [(int, int), (int, int)]> - Contains the plagiarism cases without overlapping
            res_psr <list of list of tuples (int, int)> - Contains the clusters without overlapping
    """
    #plags.sort(key = lambda tup: tup[0][0])
    if len(plags) != 0:
        plags, psr = map(list, zip(*sorted(zip(plags, psr), key = lambda tup: tup[0][0][0])))
    res_plags = []
    res_psr = []
    flag = 0
    i = 0
    while i < len(plags):
        cont_ol = 0
        if flag == 0:
            for k in range(i + 1, len(plags)):
                if plags[k][0][0] - plags[i][0][1] <= 0:
                    cont_ol += 1
        else:
            for k in range(i + 1,len(plags)):
                if plags[k][0][0] - res_plags[-1][0][1] <= 0:
                    cont_ol += 1
        if cont_ol == 0:
            if flag == 0:
                res_plags.append(plags[i])
                res_psr.append(psr[i])
            else:
                flag = 0
            i += 1
        else:
            ind_max = i
            higher_sim = 0.0
            for j in range(1, cont_ol + 1):
                if flag == 0:
                    sents_i = range(plags[i][0][0], plags[i][0][1] + 1)
                    range_i = range(plags[i][1][0], plags[i][1][1] + 1)
                else:
                    sents_i = range(res_plags[-1][0][0], res_plags[-1][0][1] + 1)
                    range_i = range(res_plags[-1][1][0], res_plags[-1][1][1] + 1)
                sents_j = range(plags[i + j][0][0], plags[i + j][0][1] + 1)
                sim_i_ol = 0.0
                sim_j_ol = 0.0
                sim_i_nol = 0.0
                sim_j_nol = 0.0
                cont_ol_sents = 0
                cont_i_nol_sents = 0
                cont_j_nol_sents = 0
                for sent in sents_i:
                    sim_max = 0.0
                    for k in range_i:
                        sim = cosine_measure(susp_bow[sent], src_bow[k])
                        if sim > sim_max:
                            sim_max = sim
                    if sent in sents_j:
                        sim_i_ol += sim_max
                        cont_ol_sents += 1
                    else:
                        sim_i_nol += sim_max
                        cont_i_nol_sents += 1
                range_j = range(plags[i + j][1][0], plags[i + j][1][1] + 1)
                for sent in sents_j:
                    sim_max = 0.0
                    for k in range_j:
                        sim = cosine_measure(susp_bow[sent], src_bow[k])
                        if sim > sim_max:
                            sim_max = sim
                    if sent in sents_i:
                        sim_j_ol += sim_max
                    else:
                        sim_j_nol += sim_max
                        cont_j_nol_sents += 1
                sim_i = sim_i_ol / cont_ol_sents
                if cont_i_nol_sents != 0:
                    sim_i = sim_i + (1 - sim_i) * sim_i_nol / float(cont_i_nol_sents)
                sim_j = sim_j_ol / cont_ol_sents
                if cont_j_nol_sents !=0 :
                    sim_j = sim_j + (1 - sim_j) * sim_j_nol / float(cont_j_nol_sents)
                if sim_i > 0.99 and sim_j > 0.99:
                    if len(sents_j) > len(sents_i):
                        if sim_j > higher_sim:
                            ind_max = i + j
                            higher_sim = sim_j
                    else:
                        if sim_i > higher_sim:
                            ind_max = i
                            higher_sim = sim_i
                elif sim_j > sim_i:
                    if sim_j > higher_sim:
                        ind_max = i + j
                        higher_sim = sim_j
                    elif sim_i > higher_sim:
                        ind_max = i
                        higher_sim = sim_i
            if flag == 0:
                res_plags.append(plags[ind_max])
                res_psr.append(psr[ind_max])
            elif ind_max != i:
                del res_plags[-1]
                del res_psr[-1]
                res_plags.append(plags[ind_max])
                res_psr.append(psr[ind_max])
            i = i + cont_ol
            flag = 1
    return res_plags, res_psr

def remove_small_plags(plags, psr, src_offsets, susp_offsets, th):
    """
    DESCRIPTION: Remove the plagiarism cases that have less tha th characters either in the source or suspicios fragments  
    INPUT: plags <list of list of two tuples [(int, int), (int, int)]> - Have the plagiarism cases represented by min and max sentence index in suspicious and source document respectively
           psr <list of list of tuples (int, int)> - Contains the clusters
           src_offsets <list of tuples (int, int)> - Contain the char offset and length of each source document sentence
           susp_offsets <list of tuples (int, int)> - Contain the char offset and length of each suspicious document sentence
    OUTPUT: res_plags <list of list of two tuples [(int, int), (int, int)]> - Contains the plagiarism cases without short cases
            res_psr <list of list of tuples (int, int)> - Contains the clusters without short cases
    """
    res_plags = []
    res_psr = []
    for i in range(len(plags)):
        arg1 = (susp_offsets[plags[i][0][0]][0], susp_offsets[plags[i][0][1]][0] + susp_offsets[plags[i][0][1]][1])
        arg2 = (src_offsets[plags[i][1][0]][0], src_offsets[plags[i][1][1]][0] + src_offsets[plags[i][1][1]][1])
        if arg1[1] - arg1[0] >= th and arg2[1] - arg2[0] >= th: 
            res_plags.append(plags[i])
            res_psr.append(psr[i])
    return res_plags, res_psr

def word_span_tokenizer(text):
    """
    DESCRIPTION: Tokenize a text in words  
    INPUT: text <string> - Text to be tokenized
    OUTPUT: words <list> - List of words from text
            offsets <list of tuple (int, int)> - Initial and final position of each word 
    """
    words = []
    offsets = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    word_detector = nltk.TreebankWordTokenizer()
    punctuation = string.punctuation
    for span in sent_detector.span_tokenize(text):
        sent = text[span[0]:span[1]].lower()
        #sent_words = [x for x in word_detector.tokenize(sent) if x[0].isalnum() and len(x) > 2]
        #sent_words = [x for x in word_detector.tokenize(sent)]
        sent_words = []
        for token in word_detector.tokenize(sent):
            for char in token:
                if char not in punctuation:
                    sent_words.append(token)
                    break
        idx = 0
        for word in sent_words:
            words.append(word) 
            pos = sent[idx:].find(word)
            #print pos
            offsets.append([span[0] + idx + pos, idx + span[0] + pos + len(word)]) #(Initial position, Final position)
            if idx == 0:#changing first word offset
                offsets[-1][0] = span[0]
            idx = idx + pos + len(word)
        if len(words) > 0:#Changing last word offset
            offsets[-1][1] = span[1]
    return words, offsets

def longest_common_substring_all(s1, s1_off, s2, s2_off, th):#Using Dynamic programming #Necesito encontrar todos los elementos mayor a un umbral en lugar de solo el mayor
    """
    DESCRIPTION: Find the common subtrings using dynamic programming  
    INPUT: s1 <list> - List of words from text 1
           s1_off <list of tuple (int, int)> - List of offsets of text1
           s2 <list> - List of words from text 2
           s2_off <list of tuple (int, int)> - List of offsets of text2
           th <int> - Threshold in characters of shortest common substring allowed
    OUTPUT: res <list tuples (int, int, int, int)> - Common subtring correspondence in text1 and text2 represented as char offsets (t1_init, t1_end, t2_init, t2_end)  
    """
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    res = []
    longest, x_longest, y_longest = 0, 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
                    y_longest = y
            else:
                m[x][y] = 0
                if m[x - 1][y - 1] != 0: 
                    len_plag = s1_off[x - 2][1] - s1_off[x - 1 - m[x - 1][y - 1]][0]
                    if len_plag > th:
                        res.append((s1_off[x - 1 - m[x - 1][y - 1]][0], s1_off[x - 2][1], s2_off[y - 1 - m[x - 1][y - 1]][0], s2_off[y - 2][1]))
        if m[x][y] != 0:#Last column
            len_plag = s1_off[x - 1][1] - s1_off[x - m[x][y]][0]
            if len_plag > th:
                res.append((s1_off[x - m[x][y]][0], s1_off[x - 1][1], s2_off[y - m[x][y]][0], s2_off[y - 1][1]))
    for y in xrange(1, len(s2)):#Last row
        if m[-1][y] != 0:
            len_plag = s1_off[-1][1] - s1_off[len(s1_off) - m[-1][y]][0]
            if len_plag > th:
                res.append((s1_off[len(s1_off) - m[-1][y]][0], s1_off[- 1][1], s2_off[y - m[-1][y]][0], s2_off[y - 1][1]))
    #return s1[x_longest - longest: x_longest]
    return res

def common_substring_pro_all(str1, str2, th_acc):
    """
    DESCRIPTION: Find the common substrings longer than some threshold  
    INPUT: str1 <list> - Text 1
           str2 <list> - Text 2
           th_acc <int> - Threshold in characters of shortest common substring allowed
    OUTPUT: res <list tuples (int, int, int, int)> - Common subtring correspondence in text1 and text2 represented as char offsets (t1_init, t1_end, t2_init, t2_end)
    """
    X, X_off = word_span_tokenizer(str1)
    Y, Y_off = word_span_tokenizer(str2)
    res = longest_common_substring_all(X, X_off, Y, Y_off, th_acc)
    return res

def verbatim_det_lcs_all(plags, psr, susp_text, src_text, susp_offsets, src_offsets, th_shortest):
    """
    DESCRIPTION: Uses longest common substring algorithm to classify a pair of documents being compared as verbatim plagarism candidate (the pair of documents), and removing the none verbatim cases if positive  
    INPUT: plags <list of list of two tuples [(int, int), (int, int)]> - Have the plagiarism cases represented by min and max sentence index in suspicious and source document respectively
           psr <list of list of tuples (int, int)> - Contains the clusters
           susp_text <string> - Suspicios document text
           src_text <string> - Source document text
           susp_offsets <list of tuples (int, int)> - Contain the char offset and length of each suspicious document sentence
           src_offsets <list of tuples (int, int)> - Contain the char offset and length of each source document sentence
           th_shortest <int> - Threshold in characters of shortest common substring allowed
    OUTPUT: res_plags <list of list of two tuples [(int, int), (int, int)]> - Contains the plagiarism cases as common substrings or the same as the arguments depending on type_plag
            res_psr <list of list of tuples (int, int)> - Contains the clusters with seeds present in the common substrings, or the same as the arguments depending on type_plag
            type_plag <0 or 1> - 1: verbatim plagiarism case    0: Other plagiarism case 
            res_long_frag <list> - Contains the lengths of common substrings
    """
    #plags   [[(susp_ini, susp_end), (src_ini, src_end)], ...]
    res_plags = []
    res_psr = []
    res_long_frag = []
    i = 0
    type_plag = 0 #0: Unknown, 1: no-obfuscation
    #print 'Plags:', len(plags)
    while i  < len(plags): #For each plagiarism case
        #print 'Case',i
        #print 'Plag case', plags[i]
        #print 'Seeds', psr[i]
        #sentences in seeds an those not in seeds
        res2 = common_substring_pro_all(susp_text[susp_offsets[plags[i][0][0]][0] : susp_offsets[plags[i][0][1]][0] + susp_offsets[plags[i][0][1]][1]], src_text[src_offsets[plags[i][1][0]][0] : src_offsets[plags[i][1][1]][0] + src_offsets[plags[i][1][1]][1]], th_shortest)
        res = []
        #Remove overlapping
        for tup_i in res2:
            flag = 0
            for tup_j in res2:
                if tup_i != tup_j and tup_i[2] >= tup_j[2] and tup_i[3] <= tup_j[3]:
                    flag = 1
                    break
            if flag == 0:
                res.append(tup_i)  
         
        #print 'Res2', res2
        #print 'Res', res
        #max_len = max([res[1] - res[0], res[3] - res[2]])
        #max_len = [(x[1] - x[0], x[3] - x[2]) for x in res]
        if len(res) > 0:
            if type_plag == 1:
                #print max_len, True, 'Removing seeds with lcs shorter than', th_shortest
                for sub_case in res:
                    res_plags.append([(susp_offsets[plags[i][0][0]][0] + sub_case[0], susp_offsets[plags[i][0][0]][0] + sub_case[1]), (src_offsets[plags[i][1][0]][0] + sub_case[2], src_offsets[plags[i][1][0]][0] + sub_case[3])])
                    res_psr.append(psr[i])
                    res_long_frag.append(max([sub_case[1] - sub_case[0], sub_case[3] - sub_case[2]]))
            else:
                #print max_len, 'Type 02-no-obfuscation detected. Starting over!'
                #print max_len, 'Type 02-no-obfuscation detected. Removing previously added cases!'
                type_plag = 1
                res_plags = []
                res_psr = []
                res_long_frag = []
                for sub_case in res:
                    res_plags.append([(susp_offsets[plags[i][0][0]][0] + sub_case[0], susp_offsets[plags[i][0][0]][0] + sub_case[1]), (src_offsets[plags[i][1][0]][0] + sub_case[2], src_offsets[plags[i][1][0]][0] + sub_case[3])])
                    res_psr.append(psr[i])
                    res_long_frag.append(max([sub_case[1] - sub_case[0], sub_case[3] - sub_case[2]]))
                #i = -1
        else:
            if type_plag != 1:
                #print max_len, False, 'Adding'
                res_plags.append(plags[i])
                res_psr.append(psr[i])
                res_long_frag.append(-1)
            #else:
                #print max_len, False, 'Removing case because 02-no-obfuscation was detected'
        i += 1
    return res_plags, res_psr, type_plag, res_long_frag

def char_preprocess(texto, chars_inserted = []):
    """
    DESCRIPTION: Normalizes some characters and adds . to headings
    INPUT: texto <string> - Text to be treated
           chars_inserted <list> - Positions of the chars inserted in the text
    OUTPUT: Returns the processed text
    """
    #'.' = 46
    #'\t' = 9
    #'\n' = 10
    #'\v' = 11
    #'\f' = 12
    #'\r' = 13
    #' ' = 32
    #'\xc2\xa0' = 160
    text = list(texto)
    newline = [10, 11, 13]
    spaces = [9, 11, 32, 160]
    last_ch = 0
    nl_flag = 0
    nl_pos = -1
    len_text = len(text)
    last_ch_pos = 0
    for i in range(len(text) - 1):
        val = ord(text[i])
        #print val, chr(val)
        #if val ==  0: #Null character
        #    text[i] = ' '
        if val == 160:
            if i - 1 >= 0 and ord(text[i - 1]) in spaces: 
                text[i - 1] = '.'
                text[i] = ' '
            elif i + 1 <= len_text and ord(text[i + 1]) in spaces:
                text[i + 1] = ' '
                text[i] = '.'
            else:
                text[i] = ' '
        elif val in newline and last_ch != 46:
            nl_flag = 1
            nl_pos = i
        elif val <= 32:
            text[i] = ' '
        else:
            if text[i].isalnum():
                if nl_flag == 1 and val >= 41 and val <= 90:#Upper case
                    text[nl_pos] = '.'
                    #text[last_ch_pos + 1] = '.'
                    if ord(text[nl_pos + 1]) not in spaces:
                        text.insert(nl_pos + 1, ' ')
                        chars_inserted.append(nl_pos + 1)
                    nl_flag = 0
                else:
                    nl_flag = 0
            last_ch = ord(text[i])
            last_ch_pos = i   
    return ''.join(text)

def update_offsets(offsets, chr_in):
    """
    DESCRIPTION: Updates the offsets of sentences after tokenize() and char_preprocess() 
    INPUT: offsets <list of tuple (int, int)> - Offsets affected by char_preprocess()
           chars_in <list> - Positions of the chars inserted in the text
    OUTPUT: Returns the corresponding orginal offsets
    """
    i = 0
    j = 0
    dec = 0
    while i < len(offsets):
        if j < len(chr_in):
            if chr_in[j] <= offsets[i][0]:
                dec += 1
                j += 1
        offsets[i][0] = offsets[i][0] - dec
        i += 1
    return dec

def tokenize(text, voc = {}, offsets = [], sents = [], rem_sw = 0):
    """
    DESCRIPTION: Tokenization and vectorization of sentences in a document  
    INPUTS: text <string> - Text to be pre-processed
            voc <dictionary> - The keys are the types (vocabulary) in a document while the values are the sentence frequency
            offsets <2-tuple> - First value contains the offset character and the next contain the length in characters
            sents <list> - Sentences of the text without tokenization
            rem_sw <integer> - Option about treatment of stopwords (0): None stopword remove, (1): 50 more common stopwords removed, (other): All stopwords removed
    OUTPUT: sent_vects <list of dictionaries> - List of dictionaries representing each sentence vector. Sparce bag of words. Also modify sents, offsets and voc.
    NOTE: If char_preprocess() is used, you must use update_offsets() also  
    """ 
    text = text.replace(chr(0), ' ')
    #chr_in = [] 
    #text = char_preprocess(text, chr_in)
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    word_detector = nltk.TreebankWordTokenizer()
    #stemmer = nltk.stem.porter.PorterStemmer()
    stemmer = Stemmer.Stemmer('english')
    sent_spans = sent_detector.span_tokenize(text)
    sent_vects = []
    if rem_sw == 0:
        stopwords = []
    elif rem_sw == 1:
        stopwords = ['the','of','and','a','in','to','is','was','it','for','with','he','be','on','i','that','by','at','you','\'s','are','not','his','this','from','but','had','which','she','they','or','an','were','we','their','been','has','have','will','would','her','n\'t','there','can','all','as','if','who','what','said']
    else:
        stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', '\'s', 'n\'t', 'can', 'will', 'just', 'don', 'should', 'now']
    for span in sent_spans: #For each sentence
        sents.append(text[span[0] : span[1]].lower())
        sent_dic = {}
        for word in word_detector.tokenize(sents[-1]): #for each word in the sentence
            if word[0].isalnum() and len(word) > 2:
                if word not in stopwords:
                    word_pp = stemmer.stemWord(word)# if word not in stopwords_all else word###Highest time consuming
                else:
                    word_pp = word
            else:
                continue
            if sent_dic.has_key(word_pp):
                sent_dic[word_pp] += 1
            else:
                sent_dic[word_pp] = 1
                if voc.has_key(word_pp):
                    voc[word_pp] += 1
                else:
                    voc[word_pp] = 1
        if len(sent_dic) > 0:
            sent_vects.append(sent_dic)
            offsets.append([span[0], span[1] - span[0]])
    #update_offsets(offsets, chr_in)
    return sent_vects

"""
MAIN CLASS
"""
class SGSPLAG:
    def __init__(self, susp_text, src_text, parameters):
        """ Parameters. """
        self.th1 = parameters['th1']
        self.th2 = parameters['th2']
        self.th3 = parameters['th3']
        self.src_gap = parameters['src_gap']
        self.src_gap_least = parameters['src_gap_least']
        self.susp_gap = parameters['susp_gap']
        self.susp_gap_least = parameters['susp_gap_least']
        self.src_size = parameters['src_size']
        self.susp_size = parameters['susp_size']
        self.min_sentlen = parameters['min_sentlen']
        self.min_plaglen = parameters['min_plaglen']
        self.rssent = parameters['rssent']
        self.tf_idf_p = parameters['tf_idf_p']
        self.rem_sw = parameters['rem_sw']
        self.verbatim_minlen = parameters['verbatim_minlen']
        self.verbatim = parameters['verbatim']
        self.summary = parameters['summary']
        self.src_gap_summary = parameters['src_gap_summary']
        self.susp_gap_summary = parameters['src_gap_summary']
        
        self.susp_text = susp_text
        self.src_text = src_text
        self.src_voc = {}
        self.susp_voc = {}
        self.src_offsets = []
        self.susp_offsets = []
        self.src_sents = []
        self.susp_sents = []
        self.detections = None

    def process(self):
        """
        DESCRIPTION: Process the plagiarism pipeline  
        INPUT: self <SGSPLAG object>
        OUTPUT: type_plag <int> - Verbatim plagarism flag
                summary_flag <int> - Summary plagarism flag
        """
        self.preprocess()
        self.detections, type_plag, summary_flag = self.compare()
        return type_plag, summary_flag
    
    def preprocess(self):
        """
        DESCRIPTION: Preprocess the suspicious and source document  
        INPUT: self <SGSPLAG object>
        OUTPUT: None. Gets bag of words with tf-idf, offsets and preprocess sentences
        """
        self.src_bow = tokenize(self.src_text, self.src_voc, self.src_offsets, self.src_sents, self.rem_sw)
        ss_treat(self.src_bow, self.src_offsets, self.min_sentlen, self.rssent, self.src_voc)    
        self.susp_bow = tokenize(self.susp_text, self.susp_voc, self.susp_offsets, self.susp_sents, self.rem_sw)
        ss_treat(self.susp_bow, self.susp_offsets, self.min_sentlen, self.rssent, self.susp_voc)
        #=======================================================================
        # self.src_bow_soft = copy.deepcopy(self.src_bow)
        # self.src_bow_hard = copy.deepcopy(self.src_bow)
        # self.susp_bow_soft = copy.deepcopy(self.susp_bow)
        # self.susp_bow_hard = copy.deepcopy(self.susp_bow)
        #=======================================================================
        
        if self.tf_idf_p == 1:
            #tf_idf_soft(self.src_bow_soft, self.src_voc, self.susp_bow_soft, self.susp_voc)
            #tf_idf_hard(self.src_bow_hard, self.src_voc, self.susp_bow_hard, self.susp_voc)
            tf_idf(self.src_bow, self.src_voc, self.susp_bow, self.susp_voc)
    
    def seeding(self):
        """
        DESCRIPTION: Creates the seeds from pair of sentece similarity using dice and cosine similarity   
        INPUT: self <SGSPLAG object>
        OUTPUT: ps <list of tuple (int, int, float, float)> - Seeds
        """
        ps = []
        for c in range(len(self.susp_bow)):
            for r in range(len(self.src_bow)):
                v1 = cosine_measure(self.susp_bow[c], self.src_bow[r])
                v2 = dice_coeff(self.susp_bow[c], self.src_bow[r])
                if v1 > self.th1 and v2 > self.th2:
                    ps.append((c, r, v1, v2))
        return ps
    
    def extension(self, ps):
        """
        DESCRIPTION: Adding two vectors  
        INPUT: self <SGSPLAG object>
               ps <list of tuple (int, int, float, float)> - Seeds
        OUTPUT: plags <list of list of two tuples [(int, int), (int, int)]> - Contains the plagiarism cases after validation
                psr <list of list of tuples (int, int)> - Contains the clusters after validation
                sim_frag <list of floats> - Stores the cosine similarity between source and suspicios fragments in the plagiarism cases after validation
        """
        psr = clustering(ps, self.src_offsets, self.susp_offsets, self.src_gap, self.susp_gap, self.src_size, self.susp_size, 0, 0)
        plags = []
        for psr_i in psr:
            plags.append([(min([x[0] for x in psr_i]), max([x[0] for x in psr_i])), (min([x[1] for x in psr_i]), max([x[1] for x in psr_i]))])
        temp_res = validation(plags, psr, self.src_offsets, self.susp_offsets, self.src_bow, self.susp_bow, self.src_gap, self.src_gap_least, self.susp_gap, self.susp_gap_least, self.src_size, self.susp_size, self.th3)
        if len(temp_res) == 0:
            plags, psr, sim_frag = [], [], []
        else:
            plags, psr, sim_frag = temp_res[0], temp_res[1], temp_res[2]
        return plags, psr, sim_frag
    
    def filtering(self, plags, psr):
        """
        DESCRIPTION: Filter the plagiarism cases by removing overlapping and short cases  
        INPUT: plags <list of list of two tuples [(int, int), (int, int)]> - Contains the plagiarism cases after validation
               psr <list of list of tuples (int, int)> - Contains the clusters after validation
        OUTPUT: plags <list of list of two tuples [(int, int), (int, int)]> - Contains the plagiarism cases. Also modify psr.
        """
        plags, psr = remove_overlap3(plags, psr, self.src_bow, self.susp_bow)
        plags, psr = remove_small_plags(plags, psr, self.src_offsets, self.susp_offsets, self.min_plaglen)
        
        #=======================================================================
        # plags, psr, type_plag = verbatim_det(plags, psr, self.susp_offsets,self.src_offsets, 0.9, 0.7)
        # if type_plag == 1:
        #     psr2 = []
        #     plags = []
        #     self.src_gap = 0
        #     self.src_gap_least = 0
        #     self.susp_gap = 0
        #     self.susp_gap_least = 0 
        #     for ps_tmp in psr:
        #         plags_tmp, psr_tmp = self.extension(ps_tmp)
        #         plags.extend(plags_tmp)
        #         psr2.extend(psr_tmp)
        #=======================================================================
        return plags
    
    def compare(self):
        """
        DESCRIPTION: Test a suspicious document for near-duplicate plagiarism with regards to a source document and return a feature list depending on the type_plag and summary_flag flags.  
        INPUT: self <SGSPLAG object>
        OUTPUT: detections <list> - Representation of plagairism cases before writing the xml file with require PAN format
                type_plag <int> - Verbatim flag
                summary_flag <int> - Summary flag
        """
        detections = []
        ps = self.seeding()
        plags, psr, sim_frag = self.extension(ps)
        plags = self.filtering(plags, psr)
        if self.verbatim != 0:
            plags_verbatim, psr_verbatim, type_plag, long_frag = verbatim_det_lcs_all(plags, psr, self.susp_text, self.src_text, self.susp_offsets,self.src_offsets, self.verbatim_minlen)
        else:
            type_plag = 0
        
        #REMOVE WHEN USING META-CLASSIFIER
        
        #=======================================================================
        # if type_plag == 0:
        #     for plag in plags: 
        #         arg1 = (self.susp_offsets[plag[0][0]][0], self.susp_offsets[plag[0][1]][0] + self.susp_offsets[plag[0][1]][1])
        #         arg2 = (self.src_offsets[plag[1][0]][0], self.src_offsets[plag[1][1]][0] + self.src_offsets[plag[1][1]][1]) 
        #         detections.append([arg1, arg2])
        # else:
        #     for plag in plags_verbatim:
        #         arg1 = plag[0][0], plag[0][1]
        #         arg2 = plag[1][0], plag[1][1]
        #         detections.append([arg1, arg2])
        #=======================================================================
        
        
        ####META-CLASSIFIER####
        if self.summary != 0:
            self.src_gap = self.src_gap_summary
            self.susp_gap = self.susp_gap_summary
            plags2, psr2, sim_frag = self.extension(ps)
            plags2 = self.filtering(plags2, psr2)
        summary_flag = 0  
        if type_plag == 0:
            sum_src = 0
            sum_susp = 0
            if self.summary != 0: 
                for plag in plags2:
                    arg1 = (self.susp_offsets[plag[0][0]][0], self.susp_offsets[plag[0][1]][0] + self.susp_offsets[plag[0][1]][1])
                    arg2 = (self.src_offsets[plag[1][0]][0], self.src_offsets[plag[1][1]][0] + self.src_offsets[plag[1][1]][1])
                    sum_susp = sum_susp + (arg1[1] - arg1[0]);   
                    sum_src = sum_src + (arg2[1] - arg2[0]);
            if sum_src != 0 and sum_src >= 3 * sum_susp: #Summary heuristic
                summary_flag = 1
                for plag in plags2:
                    arg1 = (self.susp_offsets[plag[0][0]][0], self.susp_offsets[plag[0][1]][0] + self.susp_offsets[plag[0][1]][1])
                    arg2 = (self.src_offsets[plag[1][0]][0], self.src_offsets[plag[1][1]][0] + self.src_offsets[plag[1][1]][1])
                    detections.append([arg1, arg2])
            else:
                for plag in plags:
                    arg1 = (self.susp_offsets[plag[0][0]][0], self.susp_offsets[plag[0][1]][0] + self.susp_offsets[plag[0][1]][1])
                    arg2 = (self.src_offsets[plag[1][0]][0], self.src_offsets[plag[1][1]][0] + self.src_offsets[plag[1][1]][1])
                    detections.append([arg1, arg2])
        else:
            for plag in plags_verbatim:
                arg1 = plag[0][0], plag[0][1]
                arg2 = plag[1][0], plag[1][1]
                detections.append([arg1, arg2])
        return detections, type_plag, summary_flag
    
def read_parameters(addr):
    """
    DESCRIPTION: Read te parameter from an xml file in addr  
    INPUT: addr <string> - Path to the settings file
    OUTPUT: parameters <dictionary> - Contains the parameter name and value
    """
    parameters = {}
    tree = ET.parse(addr)
    root = tree.getroot()
    for child in root:
        if child.find('type').text == 'float':
            value =  float(child.find('value').text)
        elif child.find('type').text == 'int':
            value =  int(child.find('value').text)
        else:
            value = child.find('value').text
        parameters[child.attrib['name']] = value
    return parameters

def modify_parameters(p, parameters, addr):
    """
    DESCRIPTION: Modify the parameters that were explicitly change in the command line. Useful for multiple testing when optimizing parameters.  
    INPUT: p <list> - List or command line parameters from sys.argv[5:]
           parameters <dictionary> - Dictionary of original parameters
           addr <string> - Path to xml sttings file
    OUTPUT: Code for errors. Modify parameters
    """
    p_list = {}
    tree = ET.parse(addr)
    root = tree.getroot()
    for child in root:
        p_list[child.attrib['name']] = child.find('type').text
    if len(p) % 2 != 0:
        print 'Parameter Value inconsistency'
        exit()
    for i in range(0, len(p), 2):
        if p_list.has_key(p[i]):
            #if p_list[p[i]] == type(p[i + 1]).__name__:
            if type(p[i + 1]).__name__ == 'str':
                if p_list[p[i]] == 'float':
                    val = float(p[i + 1])
                elif p_list[p[i]] == 'int':
                    val = int(p[i + 1])
                else:
                    val = p[i + 1]
                parameters[p[i]] = val
            else:
                print p[i], 'must be', p_list[p[i]], 'instead of', type(p[i + 1]).__name__
                exit()
        else:
            print 'Parameter ***', p[i],'*** not recognized!'
            print 'Use any of this parameters: ', p_list
            exit()
    return 0

def read_document(addr, encoding = 'utf-8'):
    """
    DESCRIPTION: Read a document with given encoding  
    INPUT: addr <string> - Path to file
           encoding <string> - encoding
    OUTPUT: text <string> - File content
    """
    fid = codecs.open(addr, 'r', encoding)
    text = fid.read()
    fid.close()
    return text

def serialize_features(susp, src, features, outdir):
    """
    DESCRIPTION: Serialize a feature list into a xml file. The xml is structured as described in http://www.webis.de/research/corpora/pan-pc-12/pan12/readme.txt. The filename will follow the naming scheme {susp}-{src}.xml and is located in the current directory.  Existing files will be overwritten.  
    INPUT: susp <string> - Filename of the suspicious document
           src <string> - Filename of the source document
           features <list of 2-tuples> - List containing feature-tuples of the form ((start_pos_susp, end_pos_susp), (start_pos_src, end_pos_src))
           outdir <string> - Contains the output directory of the xml file
    OUTPUT: None
    """
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)
    doc.createElement('feature')
    for f in features:
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(f[0][0]))
        feature.setAttribute('this_length', str(f[0][1] - f[0][0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(f[1][0]))
        feature.setAttribute('source_length', str(f[1][1] - f[1][0]))
        root.appendChild(feature)
    doc.writexml(open(outdir + susp.split('.')[0] + '-' + src.split('.')[0] + '.xml', 'w'), encoding = 'utf-8', newl = '\n')

'''
MAIN
'''
if __name__ == "__main__":
    """ Process the command line arguments. We expect four arguments: The path
    pointing to the pairs file and the paths pointing to the directories where
    the actual source and suspicious documents are located, and the directory 
    where the output will be saved.
    The file of pairs is a plain text file where each line represent a pair of
    documents <supicious-document-name.txt> <source-document-name.txt>: suspicious-document00001.txt source-document00294.txt
    """   
    if len(sys.argv) >= 5:
        t1 = time.time()
        srcdir = sys.argv[2]
        suspdir = sys.argv[3]
        outdir = sys.argv[4]
        if outdir[-1] != "/":
            outdir += "/"
        lines = open(sys.argv[1], 'r').readlines()
        parameters = read_parameters('settings.xml')
        print parameters
        out = modify_parameters(sys.argv[5:], parameters, 'settings.xml')
        print parameters
        for line in lines:
            print line
            susp, src = line.split()
            sgsplag_obj = SGSPLAG(read_document(os.path.join(suspdir, susp)), read_document(os.path.join(srcdir, src)), parameters)
            type_plag, summary_flag = sgsplag_obj.process()
            serialize_features(susp, src, sgsplag_obj.detections, outdir)
        t2 = time.time()
        print t2 - t1
    else:
        print('\n'.join(["Unexpected number of commandline arguments.", "Usage: ./pan13-plagiarism-text-alignment-example.py {pairs} {src-dir} {susp-dir} {out-dir}"]))