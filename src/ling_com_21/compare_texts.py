# from pathlib import Path
import os
import sys
from typing import List, Dict, Any

import nltk  # type: ignore

from utils import GetPosTaggedTokens, EstraiSequenzaPos, \
    count_avg_token_lenght, get_hapax_count, get_incremental_vocab_info, \
    get_dict_frequenze, get_trigrams_with_frequency, \
    get_trigrams_with_conditioned_probability, get_bigrams_with_measures, \
    get_bigrams_with_frequency, filter_NE_by_measure, \
    filter_bigrams_by_measure, EstraiBigrammiPos, filterBigramsByTokenFreq, \
    get_tokens_filterd_by_POS, filter_NE, get_all_NEs, filter_sentences, \
    get_sentences_with_markov2_probs, get_sentences_with_average_token_freq, \
    get_percentage_of_word_classes
from utils import ADJECTIVES, ADVERBS, NOUNS, \
    PROPER_NOUNS, CONTENT_WORDS, FUNCTIONAL_WORDS, PERSON_NE_CLASS


def program1_compare(
    file_analisys_info: Dict[str, Any],
    frasi,
    tokensTOT,
    pos_tagged_tokens,
):

    file_analisys_info["num_sentences"] = len(frasi)
    file_analisys_info["num_tokens"] = len(tokensTOT)

    # numero medio di token in una frase (escludendo la punteggiatura)

    listaPOS_SenzaPunteggiatura = EstraiSequenzaPos(
        pos_tagged_tokens, exclude_punctuation=True
    )
    file_analisys_info["avg_tokens_per_sentence"] = len(
        listaPOS_SenzaPunteggiatura
    ) / len(frasi)

    tokens_count, charsTOTcount, avg_chars_per_token = count_avg_token_lenght(
        pos_tagged_tokens, exclude_punctuation=True
    )
    file_analisys_info["avg_chars_per_token"] = avg_chars_per_token

    file_analisys_info["num_hapax_first_1000_tokens"] = get_hapax_count(tokensTOT, tokens_limit=1000)
    file_analisys_info["incremental_vocab_info"] = get_incremental_vocab_info(tokensTOT, corpus_lenght_increments=500)

    file_analisys_info["perc_content_words"] = get_percentage_of_word_classes(pos_tagged_tokens,
                                                                              CONTENT_WORDS)
    file_analisys_info["perc_functional_words"] = get_percentage_of_word_classes(pos_tagged_tokens,
                                                                                 FUNCTIONAL_WORDS)


def program2_extract_info(
    file_analisys_info: Dict[str, Any],
    frasi,
    tokensTOT,
    pos_tagged_tokens,
):

    #  estraete ed ordinate in ordine di frequenza decrescente, indicando anche la relativa
    # frequenza:
    # ◦ le 10 PoS (Part-of-Speech) più frequenti;
    listaPOS_inclusaPunteggiatura = EstraiSequenzaPos(
        pos_tagged_tokens, exclude_punctuation=False
    )
    k = 10
    topk_frequenzePOS = get_dict_frequenze(listaPOS_inclusaPunteggiatura, topk=k)
    file_analisys_info["most_frequent_POS"] = topk_frequenzePOS
    POSbigrams = nltk.bigrams(listaPOS_inclusaPunteggiatura)
    topk_POSbigrams = get_dict_frequenze(list(POSbigrams), topk=k)
    file_analisys_info["most_frequent_POS_bigrams"] = topk_POSbigrams
    # i 10 trigrammi di PoS più frequenti;
    POStrigrams = nltk.trigrams(listaPOS_inclusaPunteggiatura)
    topk_POStrigrams = get_dict_frequenze(list(POStrigrams), topk=k)
    file_analisys_info["most_frequent_POS_trigrams"] = topk_POStrigrams

    k2 = 20
    adjectives = get_tokens_filterd_by_POS(pos_tagged_tokens, ADJECTIVES)
    topk_adjectives = get_dict_frequenze(list(adjectives), topk=k2)
    file_analisys_info["most_frequent_adjectives"] = topk_adjectives

    adverbs = get_tokens_filterd_by_POS(pos_tagged_tokens, ADVERBS)
    topk_adverbs = get_dict_frequenze(list(adverbs), topk=k2)
    file_analisys_info["most_frequent_adverbs"] = topk_adverbs

    #  estraete ed ordinate i 20 bigrammi composti da Aggettivo e Sostantivo
    word_and_freqs = get_dict_frequenze(tokensTOT)
    adj_noun_bigrams = EstraiBigrammiPos(
        pos_tagged_tokens, wanted_POS_first=ADJECTIVES, wanted_POS_second=NOUNS
    )
    # dove ogni token ha una frequenza maggiore di 3:
    adj_noun_bigrams_filtered_by_tokfreq = filterBigramsByTokenFreq(
        adj_noun_bigrams, word_and_freqs, min_freq=4
    )

    # ◦ con frequenza massima, indicando anche la relativa frequenza;
    bigrams_with_frequency = get_bigrams_with_frequency(tokensTOT)
    topk_adj_noun_by_freq = filter_bigrams_by_measure(
        adj_noun_bigrams_filtered_by_tokfreq,
        bigrams_with_frequency,
        topk=k2,
    )
    file_analisys_info["topk_adj_noun_by_freq"] = topk_adj_noun_by_freq

    # ◦ con probabilità condizionata massima, indicando anche la relativa probabilità;
    (
        bigrams_with_LMM,
        bigrams_with_conditioned_probability,
    ) = get_bigrams_with_measures(  # get_bigrams_with_conditioned_probability(
        tokensTOT,
        word_and_freqs,
        bigrams_with_frequency,
    )
    topk_adj_noun_by_cond_prob = filter_bigrams_by_measure(
        adj_noun_bigrams_filtered_by_tokfreq,
        bigrams_with_conditioned_probability,
        topk=k2,
    )
    file_analisys_info["topk_adj_noun_by_cond_prob"] = topk_adj_noun_by_cond_prob

    # ◦ con forza associativa (calcolata in termini di Local Mutual Information) massima,
    # indicando anche la relativa forza associativa;
    topk_adj_noun_by_LMM = filter_bigrams_by_measure(
        adj_noun_bigrams_filtered_by_tokfreq,
        bigrams_with_LMM,
        topk=k2,
    )
    file_analisys_info["topk_adj_noun_by_LMM"] = topk_adj_noun_by_LMM

    NE_by_class_and_POS = get_all_NEs(pos_tagged_tokens)
    selected_NE_list = filter_NE(
        NE_by_class_and_POS,
        wanted_POSes=PROPER_NOUNS,
        wanted_NE_classes=[PERSON_NE_CLASS],
    )
    k3 = 15
    file_analisys_info["selected_topk_NE_with_freq"] = filter_NE_by_measure(
        selected_NE_list, word_and_freqs, topk=k3
    )

    #  estraete le frasi con almeno 6 token e più corta di 25 token,
    # dove ogni singolo token occorre almeno due volte nel corpus di riferimento:
    filtered_sentences = filter_sentences(
        frasi,
        word_and_freqs,
        min_lenght=6,
        max_lenght=24,
        min_token_freq=2,
    )

    # ◦ con la media della distribuzione di frequenza dei token più alta, in un caso, e più bassa
    # nell’altro, riportando anche la distribuzione media di frequenza. La distribuzione media
    # di frequenza deve essere calcolata tenendo in considerazione la frequenza di tutti i token
    # presenti nella frase (calcolando la frequenza nel corpus dal quale la frase è stata estratta)
    # e dividendo la somma delle frequenze per il numero di token della frase stessa;
    sentences_with_average_token_freq = get_sentences_with_average_token_freq(
        filtered_sentences,
        word_and_freqs,
    )
    text_of_top_sentence_for_average_token_freq = list(
        sentences_with_average_token_freq
    )[0]
    prob_of_top_sentence_for_average_token_freq = sentences_with_average_token_freq[
        text_of_top_sentence_for_average_token_freq
    ]
    top_sentence_for_average_token_freq = (
        text_of_top_sentence_for_average_token_freq,
        prob_of_top_sentence_for_average_token_freq,
    )
    file_analisys_info[
        "top_sentence_for_average_token_freq"
    ] = top_sentence_for_average_token_freq

    text_of_last_sentence_for_average_token_freq = list(
        sentences_with_average_token_freq
    )[-1]
    prob_of_last_sentence_for_average_token_freq = sentences_with_average_token_freq[
        text_of_last_sentence_for_average_token_freq
    ]
    last_sentence_for_average_token_freq = (
        text_of_last_sentence_for_average_token_freq,
        prob_of_last_sentence_for_average_token_freq,
    )
    file_analisys_info[
        "last_sentence_for_average_token_freq"
    ] = last_sentence_for_average_token_freq

    # ◦ con probabilità più alta, dove la probabilità deve essere calcolata attraverso un modello
    # di Markov di ordine 2. Il modello deve usare le statistiche estratte dal corpus che
    # contiene le frasi;
    trigrams_with_frequency = get_trigrams_with_frequency(tokensTOT)
    trigrams_with_conditioned_probability = get_trigrams_with_conditioned_probability(
        tokensTOT,
        bigrams_with_frequency,
        trigrams_with_frequency,
    )

    sentences_with_markov2_probs = get_sentences_with_markov2_probs(
        filtered_sentences,
        word_and_freqs,
        len(tokensTOT),
        bigrams_with_conditioned_probability,
        trigrams_with_conditioned_probability,
    )

    top_sentence_for_prob = list(sentences_with_markov2_probs)[0]
    prob_of_top_sentence_for_prob = sentences_with_markov2_probs[top_sentence_for_prob]
    max_mkv2_prob_sentence = (top_sentence_for_prob, prob_of_top_sentence_for_prob)
    file_analisys_info["max_mkv2_prob_sentence"] = max_mkv2_prob_sentence


def getFileAnalisysInfo(filepath: str) -> Dict:
    with open(filepath, mode="r", encoding="utf-8") as fileInput:
        raw = fileInput.read()

    #  il numero di frasi e di token:
    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    frasi: List[str] = sent_tokenizer.tokenize(raw)
    tokensTOT, pos_tagged_tokens = GetPosTaggedTokens(frasi)

    file_analisys_info: Dict[str, Any] = dict()
    file_analisys_info["filename"] = os.path.basename(filepath)

    program1_compare(file_analisys_info, frasi, tokensTOT, pos_tagged_tokens)
    program2_extract_info(file_analisys_info, frasi, tokensTOT, pos_tagged_tokens)

    return file_analisys_info


def print_results_helper_pt1(file_analisys_info1, file_analisys_info2):

    filename1 = file_analisys_info1["filename"]
    filename2 = file_analisys_info2["filename"]

    print(
        f"Numero di frasi: "
        f"{file_analisys_info1['num_sentences']} ({filename1}) e "
        f"{file_analisys_info2['num_sentences']} ({filename2})."
    )

    print(
        f"Numero di token totali: "
        f"{file_analisys_info1['num_tokens']} ({filename1}) e "
        f"{file_analisys_info2['num_tokens']} ({filename2})."
    )
    # I due testi hanno rispettivamente 364 e 567 frasi.
    # I due testi hanno rispettivamente 10339 e 6462 token totali.

    print(
        f"Numero medio di token in una frase (escludendo la punteggiatura): "
        f"{file_analisys_info1['avg_tokens_per_sentence']:.2f} ({filename1}) e "
        f"{file_analisys_info2['avg_tokens_per_sentence']:.2f} ({filename2})."
    )

    print(
        f"Numero medio dei caratteri di un token (escludendo la punteggiatura): "
        f"{file_analisys_info1['avg_chars_per_token']:.2f} ({filename1}) e "
        f"{file_analisys_info2['avg_chars_per_token']:.2f} ({filename2})."
    )

    print(
        f"Numero di hapax sui primi 1000 token: "
        f"{file_analisys_info1['num_hapax_first_1000_tokens']} ({filename1}) e "
        f"{file_analisys_info2['num_hapax_first_1000_tokens']} ({filename2})."
    )

    print(f"La grandezza del vocabolario e la ricchezza lessicale (Type Token Ratio, TTR):")
    for file_analisys_info in [file_analisys_info1, file_analisys_info2]:
        print(f"{filename1} :")
        for corpus_limit in file_analisys_info["incremental_vocab_info"]:
            vocab_size = file_analisys_info["incremental_vocab_info"][corpus_limit]["vocab_size"]
            TTR = file_analisys_info["incremental_vocab_info"][corpus_limit]["TTR"]
            print(f"Corpus lenght: {corpus_limit}, vocab_size: {vocab_size}, TTR: {TTR}")

    print(
        f"Percentuale delle parole piene (Aggettivi, Sostantivi, Verbi, Avverbi) : "
        f"{file_analisys_info1['perc_content_words']:.2%} ({filename1}) e "
        f"{file_analisys_info2['perc_content_words']:.2%} ({filename2})."
    )
    print(
        f"Percentuale delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi) : "
        f"{file_analisys_info1['perc_functional_words']:.2%} ({filename1}) e "
        f"{file_analisys_info2['perc_functional_words']:.2%} ({filename2})."
    )

def print_results_helper_pt2(file_analisys_info1, file_analisys_info2):

    file_infos = [file_analisys_info1, file_analisys_info2]
    print(f"Le 10 PoS (Part-of-Speech) più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_POS", "POS")
    print(f"I 10 bigram di PoS più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_POS_bigrams", "POS Bigram")
    print(f"I 10 trigrammi di PoS più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_POS_trigrams", "POS Trigram")
    print(f"I 20 aggettivi più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_adjectives", "Adj")
    print(f"I 20 avverbi più frequenti sono:")
    print_info_helper(file_infos, "most_frequent_adverbs", "Adv")

    print(
        f"I 20 bigrammi composti da Aggettivo e Sostantivo "
        f"(dove ogni token ha una frequenza maggiore di 3), "
        f"con frequenza massima, sono:"
    )
    print_info_helper(file_infos, "topk_adj_noun_by_freq", "Bigram", measure="freq")

    print(
        f"I 20 bigrammi composti da Aggettivo e Sostantivo "
        f"(dove ogni token ha una frequenza maggiore di 3), "
        f"con probabilità condizionata massima, sono:"
    )
    print_info_helper(
        file_infos, "topk_adj_noun_by_cond_prob", "Bigram", measure="prob.cond"
    )

    print(
        f"I 20 bigrammi composti da Aggettivo e Sostantivo "
        f"(dove ogni token ha una frequenza maggiore di 3), "
        f"con Local Mutual Information (LMM) massima, sono:"
    )
    print_info_helper(file_infos, "topk_adj_noun_by_LMM", "Bigram", measure="LMM")

    print(
        f"Tra le frasi con almeno 6 token e più corte di 25 token, "
        f"di cui ogni singolo token occorre almeno due volte nel corpus di riferimento, "
        f"vi sono:"
    )

    print(f"Con la media della distribuzione di frequenza dei token più alta:")
    for file_info in file_infos:
        print(f"{file_info['filename']}: ")
        top_sentence_for_average_token_freq = file_info[
            "top_sentence_for_average_token_freq"
        ]
        print(
            f"Avg. freq.: {top_sentence_for_average_token_freq[1]} testo: {top_sentence_for_average_token_freq[0]}"
        )

    print(f"Con la media della distribuzione di frequenza dei token più bassa:")
    for file_info in file_infos:
        print(f"{file_info['filename']}: ")
        last_sentence_for_average_token_freq = file_info[
            "last_sentence_for_average_token_freq"
        ]
        print(
            f"Avg. freq.: {last_sentence_for_average_token_freq[1]} testo: {last_sentence_for_average_token_freq[0]}"
        )

    print(f"con con probabilità più alta secondo un modello di Markov di ordine 2:")
    for file_info in file_infos:
        print(f"{file_info['filename']}: ")
        max_mkv2_prob_sentence = file_info["max_mkv2_prob_sentence"]
        print(f"Prob: {max_mkv2_prob_sentence[1]} testo: {max_mkv2_prob_sentence[0]}")

    print(
        f"15 nomi propri di persona più frequenti (tipi), ordinati per frequenza, sono:"
    )
    print_info_helper(file_infos, "selected_topk_NE_with_freq", "NE", measure="freq")


def analize_files_and_print_results(filepath1: str, filepath2: str):
    print(f"Caricamento dei file {filepath1} e {filepath2}")

    file_analisys_info1 = getFileAnalisysInfo(filepath1)
    file_analisys_info2 = getFileAnalisysInfo(filepath2)
    filename1 = file_analisys_info1["filename"]
    filename2 = file_analisys_info2["filename"]
    print(f"Analisi dei due testi {filename1} e {filename2} :")

    print_results_helper_pt1(file_analisys_info1, file_analisys_info2)
    print_results_helper_pt2(file_analisys_info1, file_analisys_info2)

    # TODO: output: file di testo contenenti l'output ben formattato dei programmi.


def print_info_helper(
    file_infos, elements_key: str, element_descr: str, measure="freq"
):
    for file_info in file_infos:
        print(f"{file_info['filename']}: ")
        for element_with_freq in file_info[elements_key].items():
            print(
                f"{element_descr}: {element_with_freq[0]}  ----{measure}: {element_with_freq[1]:.2f}"
            )


def main():
    if len(sys.argv) >= 3:
        filepath1 = sys.argv[1]
        filepath2 = sys.argv[2]
    else:
        filepath1 = "..\\..\\Cablegate.txt"
        filepath2 = "..\\..\\Colbert.txt"

    analize_files_and_print_results(
        filepath1=filepath1,
        filepath2=filepath2,
    )


if __name__ == "__main__":
    main()
