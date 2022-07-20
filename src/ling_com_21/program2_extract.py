import os
import sys
from typing import Dict, Any

import nltk  # type: ignore

from utils import (
    EstraiSequenzaPos,
    get_dict_frequenze,
    get_tokens_filterd_by_POS,
    ADJECTIVES,
    ADVERBS,
    EstraiBigrammiPos,
    NOUNS,
    filterBigramsByTokenFreq,
    get_bigrams_with_frequency,
    filter_bigrams_by_measure,
    get_bigrams_with_measures,
    get_all_NEs,
    filter_NE,
    PROPER_NOUNS,
    PERSON_NE_CLASS,
    filter_NE_by_measure,
    filter_sentences,
    get_sentences_with_average_token_freq,
    get_trigrams_with_frequency,
    get_trigrams_with_conditioned_probability,
    get_sentences_with_markov2_probs,
    print_info_helper,
    analize_files_and_print_results, get_basic_file_info,
)


def program2_extract_info(
    filepath: str,
) -> Dict[str, Any]:

    file_analisys_info: Dict[str, Any] = dict()
    file_analisys_info["filename"] = os.path.basename(filepath)
    frasi, tokensTOT, pos_tagged_tokens = get_basic_file_info(filepath)

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

    return file_analisys_info


def print_results_helper_pt2(file_analisys_info1, file_analisys_info2):

    filename1 = file_analisys_info1["filename"]
    filename2 = file_analisys_info2["filename"]

    print(f"Analisi dei due testi {filename1} e {filename2} :")

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
        extraction_function=program2_extract_info,
        output_function=print_results_helper_pt2,
    )


if __name__ == "__main__":
    main()
