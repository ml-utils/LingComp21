# from pathlib import Path
import os
import sys
from typing import List, Tuple, Dict, Any, Optional

import nltk

ALL_PUNKTUATION = [".", ",", ":", "(", ")"] # "SYM" (todo: verify what symbols include)
ADJECTIVES = ["JJ", "JJR", "JJS"]

def EstraiFrasi(filepath: str) -> List[str]:

    with open(filepath, mode="r", encoding="utf-8") as fileInput1:
        raw1 = fileInput1.read()
    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    return sent_tokenizer.tokenize(raw1)

# slides name "AnnotazioneLinguistica"
def GetPosTaggedTokens(frasi: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    tokensTOT: List[str] = []
    pos_tagged_tokens: List[Tuple[str, str]] = []
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensTOT += tokens
        pos_tagged_tokens_this_sentence = nltk.pos_tag(tokens)
        pos_tagged_tokens += pos_tagged_tokens_this_sentence
    return tokensTOT, pos_tagged_tokens

def EstraiSequenzaPos(pos_tagged_tokens: List[Tuple[str, str]], exclude_punctuation=False) -> List[str]:
    listaPOS: List[str] = []
    TOKEN_IDX = 0
    POS_IDX = 1
    for pos_taggged_token in pos_tagged_tokens:
        if exclude_punctuation and pos_taggged_token[POS_IDX] in ALL_PUNKTUATION:
            continue
        else:
            listaPOS.append(pos_taggged_token[POS_IDX])
    return listaPOS

def count_avg_tokens_per_sentence(frasi: List[str], exclude_punctuation=True):
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        pos_tagged_tokens_this_sentence = nltk.pos_tag(tokens)

def count_avg_token_lenght(pos_tagged_tokens: List[Tuple[str, str]], exclude_punctuation: bool=True):
    TOKEN_IDX = 0
    POS_IDX = 1
    tokens_count = 0
    charsTOTcount = 0
    for pos_taggged_token in pos_tagged_tokens:
        if exclude_punctuation and pos_taggged_token[POS_IDX] in ALL_PUNKTUATION:
            continue
        else:
            tokens_count += 1
            charsTOTcount += len(pos_taggged_token[TOKEN_IDX])

    avg = charsTOTcount / tokens_count
    return tokens_count, charsTOTcount, avg


def get_dict_frequenze(
        mylist: List[Any],
        sorted:bool=True,
        topk: Optional[int]=None
) -> Dict[Any, int]:

    freqDistribution = nltk.FreqDist(mylist)
    topk_elements = freqDistribution.most_common(topk)
    topk_elements_as_dict = {x[0]: x[1] for x in topk_elements}

    return topk_elements_as_dict


def get_dict_frequenze_POS(listaPOS: List[str], sorted=True, topk=None) -> Dict[str, int]:

    return get_dict_frequenze(listaPOS, sorted=sorted, topk=topk)


def get_tokens_filterd_by_POS(
        pos_tagged_tokens: List[Tuple[str, str]],
        wanted_POS: List[str]
) -> List[str]:

    TOK_IDX = 0
    POS_IDX = 1

    POS_list = []
    for pos_tagged_token in pos_tagged_tokens:
        if pos_tagged_token[POS_IDX] in wanted_POS:
            POS_list += pos_tagged_token[TOK_IDX]

    return POS_list


def getFileAnalisysInfo(filepath: str) -> Dict:
    with open(filepath, mode='r', encoding="utf-8") as fileInput:
        raw = fileInput.read()

    #  il numero di frasi e di token:
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi = sent_tokenizer.tokenize(raw)
    tokensTOT = getWordTokenized(frasi)
    file_analisys_info = dict()
    file_analisys_info["filename"] = os.path.basename(filepath)
    file_analisys_info["num_sentences"] = len(frasi)
    file_analisys_info["num_tokens"] = len(tokensTOT)

    # numero medio di token in una frase (escludendo la punteggiatura)
    tokensTOT, pos_tagged_tokens = GetPosTaggedTokens(frasi)
    listaPOS_SenzaPunteggiatura = EstraiSequenzaPos(pos_tagged_tokens, exclude_punctuation=True)
    file_analisys_info["avg_tokens_per_sentence"] = len(listaPOS_SenzaPunteggiatura) / len (frasi)

    tokens_count, charsTOTcount, avg_chars_per_token = count_avg_token_lenght(pos_tagged_tokens, exclude_punctuation=True)
    file_analisys_info["avg_chars_per_token"] = avg_chars_per_token

    #  estraete ed ordinate in ordine di frequenza decrescente, indicando anche la relativa
    # frequenza:
    # ◦ le 10 PoS (Part-of-Speech) più frequenti;
    listaPOS_inclusaPunteggiatura = EstraiSequenzaPos(pos_tagged_tokens, exclude_punctuation=False)
    k = 10
    topk_frequenzePOS = get_dict_frequenze(listaPOS_inclusaPunteggiatura, sorted=True, topk=k)
    file_analisys_info["most_frequent_POS"] = topk_frequenzePOS
    POSbigrams = nltk.bigrams(listaPOS_inclusaPunteggiatura)
    topk_POSbigrams = get_dict_frequenze(list(POSbigrams), sorted=True, topk=k)
    file_analisys_info["most_frequent_POS_bigrams"] = topk_POSbigrams
    # i 10 trigrammi di PoS più frequenti;
    POStrigrams = nltk.trigrams(listaPOS_inclusaPunteggiatura)
    topk_POStrigrams = get_dict_frequenze(list(POStrigrams), sorted=True, topk=k)
    file_analisys_info["most_frequent_POS_trigrams"] = topk_POStrigrams

    k2 = 20
    adjectives = get_tokens_filterd_by_POS(pos_tagged_tokens, ADJECTIVES)
    topk_adjectives = get_dict_frequenze(list(adjectives), sorted=True, topk=k2)
    file_analisys_info["most_frequent_adjectives"] = topk_adjectives

    return file_analisys_info

def SortDecreasing(sort_me: Dict):
    return sorted(sort_me.items(), key=lambda x: x[1], reverse=True)

def read_files(filepath1: str, filepath2: str):
    print(f"Caricamento dei file {filepath1} e {filepath2}")

    file_analisys_info1 = getFileAnalisysInfo(filepath1)
    file_analisys_info2 = getFileAnalisysInfo(filepath2)
    filename1 = file_analisys_info1['filename']
    filename2 = file_analisys_info2['filename']
    print(f"Analisi dei due testi {filename1} e {filename2} :")

    print(f"Numero di frasi: "
          f"{file_analisys_info1['num_sentences']} ({filename1}) e "
          f"{file_analisys_info2['num_sentences']} ({filename2}).")

    print(f"Numero di token totali: "
          f"{file_analisys_info1['num_tokens']} ({filename1}) e "
          f"{file_analisys_info2['num_tokens']} ({filename2}).")
    # I due testi hanno rispettivamente 364 e 567 frasi.
    # I due testi hanno rispettivamente 10339 e 6462 token totali.

    print(f"Numero medio di token in una frase (escludendo la punteggiatura): "
          f"{file_analisys_info1['avg_tokens_per_sentence']:.2f} ({filename1}) e "
          f"{file_analisys_info2['avg_tokens_per_sentence']:.2f} ({filename2}).")

    print(f"Numero medio dei caratteri di un token (escludendo la punteggiatura): "
          f"{file_analisys_info1['avg_chars_per_token']:.2f} ({filename1}) e "
          f"{file_analisys_info2['avg_chars_per_token']:.2f} ({filename2}).")

    file_infos = [file_analisys_info1, file_analisys_info2]
    print(f"Le 10 PoS (Part-of-Speech) più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_POS", "POS")
    print(f"I 10 bigram di PoS più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_POS_bigrams", "POS Bigram")
    print(f"I 10 trigrammi di PoS più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_POS_trigrams", "POS Trigram")
    print(f"I 20 Aggettivi più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_adjectives", "Adj")

    #  il numero di hapax sui primi 1000 token; (già fatto come esercizio)

    #  la grandezza del vocabolario e la ricchezza lessicale calcolata attraverso la Type Token
    # Ratio (TTR), in entrambi i casi calcolati all'aumentare del corpus per porzioni incrementali
    # di 500 token;
    # (già fatto come esercizio)

    # #  distribuzione in termini di percentuale dell’insieme delle parole piene (Aggettivi, Sostantivi,
    #     # Verbi, Avverbi) e delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi).

    # todo: calcoli su POS e frequenza, bi e tri-grammi pos

def print_info_helper(file_infos, elements_key:str, element_descr: str):
    for file_info in file_infos:
        print(f"{file_info['filename']}: ")
        for element_with_freq in file_info[elements_key].items():
            print(f"{element_descr}: {element_with_freq[0]}  ----freq: {element_with_freq[1]}")

def getWordTokenized(frasi):
    tokensTOT = []
    for frase in frasi:
        # print(f"{frase}")
        tokens = nltk.word_tokenize(frase)
        tokensTOT += tokens
    return tokensTOT


def annotate():
    pass

def compare_two_texts():
    # Confrontate i due testi sulla base delle seguenti informazioni statistiche:
    #  il numero di frasi e di token;
    #  la lunghezza media delle frasi in termini di token e dei token (escludendo la punteggiatura)
    # in termini di caratteri;
    #  il numero di hapax sui primi 1000 token;
    #  la grandezza del vocabolario e la ricchezza lessicale calcolata attraverso la Type Token
    # Ratio (TTR), in entrambi i casi calcolati all'aumentare del corpus per porzioni incrementali
    # di 500 token;
    #  distribuzione in termini di percentuale dell’insieme delle parole piene (Aggettivi, Sostantivi,
    # Verbi, Avverbi) e delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi).

    # output: file di testo contenenti l'output ben formattato dei programmi.
    pass

def extract_info_from_txts():

    # Per ognuno dei due corpora estraete le seguenti informazioni:
    #  estraete ed ordinate in ordine di frequenza decrescente, indicando anche la relativa
    # frequenza:
    # ◦ le 10 PoS (Part-of-Speech) più frequenti;
    # ◦ i 10 bigrammi di PoS più frequenti;
    # ◦ i 10 trigrammi di PoS più frequenti;
    
    # ◦ i 20 Aggettivi e i 20 Avverbi più frequenti;
    #  estraete ed ordinate i 20 bigrammi composti da Aggettivo e Sostantivo e dove ogni token ha
    # una frequenza maggiore di 3:
    # ◦ con frequenza massima, indicando anche la relativa frequenza;
    # ◦ con probabilità condizionata massima, indicando anche la relativa probabilità;
    # ◦ con forza associativa (calcolata in termini di Local Mutual Information) massima,
    # indicando anche la relativa forza associativa;
    #  estraete le frasi con almeno 6 token e più corta di 25 token, dove ogni singolo token occorre
    # almeno due volte nel corpus di riferimento:
    # ◦ con la media della distribuzione di frequenza dei token più alta, in un caso, e più bassa
    # nell’altro, riportando anche la distribuzione media di frequenza. La distribuzione media
    # di frequenza deve essere calcolata tenendo in considerazione la frequenza di tutti i token
    # presenti nella frase (calcolando la frequenza nel corpus dal quale la frase è stata estratta)
    # e dividendo la somma delle frequenze per il numero di token della frase stessa;
    # ◦ con probabilità più alta, dove la probabilità deve essere calcolata attraverso un modello
    # di Markov di ordine 2. Il modello deve usare le statistiche estratte dal corpus che
    # contiene le frasi;
    #  dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraete:
    # ◦ i 15 nomi propri di persona più frequenti (tipi), ordinati per frequenza.

    # output: file di testo contenenti l'output ben formattato dei programmi.
    pass

def main():
    if len(sys.argv) >= 3:
        filepath1 = sys.argv[1]
        filepath2 = sys.argv[2]
    else:
        filepath1 = "..\\..\\Cablegate.txt"
        filepath2 = "..\\..\\Colbert.txt"

    read_files(
        filepath1=filepath1,
        filepath2=filepath2,
    )

if __name__ == '__main__':
    main()
