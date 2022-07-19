# from pathlib import Path
import math
import os
import sys
from itertools import islice
from typing import List, Tuple, Dict, Any, Optional, Union

import nltk  # type: ignore

ALL_PUNKTUATION = [".", ",", ":", "(", ")"]  # "SYM" (todo: verify what symbols include)
ADJECTIVES = ["JJ", "JJR", "JJS"]
ADVERBS = ["RB", "RBR", "RBS", "WRB"]
NOUNS = ["NN", "NNS", "NNP", "NNPS"]  # sostantivi
PROPER_NOUNS = ["NNP", "NNPS"]
PERSON_NE_CLASS = "PERSON"  # , "GPE", "ORGANIZATION"


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


def EstraiSequenzaPos(
    pos_tagged_tokens: List[Tuple[str, str]], exclude_punctuation=False
) -> List[str]:
    listaPOS: List[str] = []
    TOKEN_IDX = 0
    POS_IDX = 1
    for pos_taggged_token in pos_tagged_tokens:
        if exclude_punctuation and pos_taggged_token[POS_IDX] in ALL_PUNKTUATION:
            continue
        else:
            listaPOS.append(pos_taggged_token[POS_IDX])
    return listaPOS


def count_avg_token_lenght(
    pos_tagged_tokens: List[Tuple[str, str]], exclude_punctuation: bool = True
):
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


def get_word_types_with_freq(
    allCorpusTokens: List[str],
) -> Dict[str, int]:

    word_types_with_freq = dict()
    word_types = set(allCorpusTokens)
    for word_type in word_types:
        tokFreq = allCorpusTokens.count(word_type)
        word_types_with_freq[word_type] = tokFreq
    return word_types_with_freq


def get_hapax_count(tokensTOT: List[str], tokens_limit: int) -> int:
    tokensTOTsliced = tokensTOT[:tokens_limit]
    word_types_with_freq = get_word_types_with_freq(tokensTOTsliced)

    hapax_list = []
    for word, freq in word_types_with_freq.items():
        if freq == 1:
            hapax_list.append(word)

    return len(hapax_list)


def get_incremental_vocab_info(
    tokensTOT: List[str],
    corpus_lenght_increments: int,
):
    incremental_vocab_info: Dict[int, Dict[str, float]] = dict()

    totalCorpusLenght = len(tokensTOT)
    # num_steps = totalCorpusLenght // corpus_lenght_increments
    range_start = min(totalCorpusLenght, corpus_lenght_increments)
    range_stop = totalCorpusLenght
    range_step = corpus_lenght_increments
    for tokens_limit in range(range_start, range_stop, range_step):
        limitedTokens = tokensTOT[:tokens_limit]
        vocab_size = len(set(limitedTokens))
        type_token_ratio = vocab_size / tokens_limit

        incremental_vocab_info[tokens_limit] = dict()
        incremental_vocab_info[tokens_limit]["vocab_size"] = vocab_size
        incremental_vocab_info[tokens_limit]["TTR"] = type_token_ratio

    return incremental_vocab_info


def get_dict_frequenze(mylist: List[Any], topk: Optional[int] = None) -> Dict[Any, int]:

    freqDistribution = nltk.FreqDist(mylist)
    topk_elements = freqDistribution.most_common(topk)
    topk_elements_as_dict = {x[0]: x[1] for x in topk_elements}

    return topk_elements_as_dict


def get_trigrams_with_frequency(
    allCorpusTokens: List[str],
) -> Dict[Tuple[str, str, str], int]:
    """

    :param allCorpusTokens:
    :return: A dictionary of trigrams (keys), sorted in decreasing order by their frequency (values).
    """
    allCorpusTrigrams: List[Tuple[str, str, str]] = list(nltk.trigrams(allCorpusTokens))
    trigrams_with_frequency: Dict[Tuple[str, str, str], int] = dict()

    uniqueTrigrams = set(allCorpusTrigrams)
    for trigramma in uniqueTrigrams:
        frequenzaTrigramma = allCorpusTrigrams.count(trigramma)
        trigrams_with_frequency[trigramma] = frequenzaTrigramma

    return SortDecreasing(trigrams_with_frequency)


def get_trigrams_with_conditioned_probability(
    allCorpusTokens: List[str],
    bigrams_with_frequency: Dict[Tuple[str, str], int],
    trigrams_with_frequency: Dict[Tuple[str, str, str], int],
) -> Dict[Tuple[str, str, str], float]:  # trigrams_with_conditioned_probability

    allCorpusTrigrams: List[Tuple[str, str, str]] = list(nltk.trigrams(allCorpusTokens))
    trigrams_with_conditioned_probability = dict()

    uniqueTrigrams = set(allCorpusTrigrams)
    for trigramma in uniqueTrigrams:
        bigramma: Tuple[str, str] = (trigramma[0], trigramma[1])
        frequenzaBigramma = bigrams_with_frequency[bigramma]
        frequenzaTrigramma = trigrams_with_frequency[trigramma]
        probCondizionataBigramma = frequenzaTrigramma * 1.0 / frequenzaBigramma * 1.0
        trigrams_with_conditioned_probability[trigramma] = probCondizionataBigramma

    return trigrams_with_conditioned_probability


def get_bigrams_with_measures(
    allCorpusTokens: List[str],
    word_types_with_freq: Dict[str, int],
    bigrams_with_frequency: Dict[Tuple[str, str], int],
    # topk: Optional[int] = None,
) -> Tuple[
    Dict[Tuple[str, str], float],  # bigrams_with_LMM
    Dict[Tuple[str, str], float],  # bigrams_with_conditioned_probability
]:  # todo: return named tuple

    # topk_POSbigrams = get_dict_frequenze(adj_noun_bigrams_filtered, topk=k2)

    # NB: conditioned probability calculated based on all bigrams in corpus, not just the filtered ones
    allCorpusBigrams = list(nltk.bigrams(allCorpusTokens))
    bigrams_with_LMM = dict()
    bigrams_with_conditioned_probability = dict()

    uniqueBigrams = set(allCorpusBigrams)
    for bigramma in uniqueBigrams:

        TOKEN_A_IDX = 0
        TOKEN_B_IDX = 1
        frequenzaBigramma = bigrams_with_frequency[bigramma]
        probBigramma = frequenzaBigramma * 1.0 / len(allCorpusBigrams) * 1.0
        frequenzaA = word_types_with_freq[bigramma[TOKEN_A_IDX]]
        frequenzaB = word_types_with_freq[bigramma[TOKEN_B_IDX]]
        el_A_prob = frequenzaA * 1.0 / len(allCorpusTokens) * 1.0
        el_B_prob = frequenzaB * 1.0 / len(allCorpusTokens) * 1.0

        bigram_MutualInformation = math.log2(probBigramma / (el_A_prob * el_B_prob))
        bigram_LocalMutualInformation = frequenzaBigramma * bigram_MutualInformation
        bigrams_with_LMM[bigramma] = bigram_LocalMutualInformation

        probCondizionataBigramma = frequenzaBigramma * 1.0 / frequenzaA * 1.0
        bigrams_with_conditioned_probability[bigramma] = probCondizionataBigramma

    bigrams_with_LMM = SortDecreasing(bigrams_with_LMM)
    bigrams_with_conditioned_probability = SortDecreasing(
        bigrams_with_conditioned_probability
    )

    return bigrams_with_LMM, bigrams_with_conditioned_probability


def get_bigrams_with_frequency(
    allCorpusTokens: List[str],
) -> Dict[Tuple[str, str], int]:
    """

    :param allCorpusTokens:
    :return: A dictionary of bigrams (keys), sorted in decreasing order by their frequency (values).
    """
    allCorpusBigrams: List[Tuple[str, str]] = list(nltk.bigrams(allCorpusTokens))
    bigrams_with_frequency: Dict[Tuple[str, str], int] = dict()

    uniqueBigrams = set(allCorpusBigrams)
    for bigramma in uniqueBigrams:
        frequenzaBigramma = allCorpusBigrams.count(bigramma)
        bigrams_with_frequency[bigramma] = frequenzaBigramma

    return SortDecreasing(bigrams_with_frequency)


def filter_NE_by_measure(
    selected_NE_list,
    word_and_freqs,  # check that this is sorted in decreasing order
    topk: int,
):
    unique_selected_NEs = set(selected_NE_list)
    unique_selected_NEs_with_freq = dict()
    for unique_selected_NE in unique_selected_NEs:
        freq = word_and_freqs[unique_selected_NE]
        unique_selected_NEs_with_freq[unique_selected_NE] = freq

    top_el = dict(
        list(islice(SortDecreasing(unique_selected_NEs_with_freq).items(), topk))
    )

    return top_el


def filter_bigrams_by_measure(
    tokpos_bigrams_to_filter: List[Tuple[Tuple[str, str], Tuple[str, str]]],  # example:
    bigrams_with_measure: Union[
        Dict[Tuple[str, str], float], Dict[Tuple[str, str], int]
    ],
    topk: int,
) -> Dict[Tuple[str, str], float]:
    """

    :param tokpos_bigrams_to_filter:
    :param bigrams_with_measure: Assumes that this dictionary is sorted in descreasing order on the measure.
    :param topk:
    :return:
    """
    TOK_IDX = 0
    bare_bigrams_to_filter = [
        (x[0][TOK_IDX], x[1][TOK_IDX]) for x in tokpos_bigrams_to_filter
    ]

    topk_bigrams_by_measure = dict()
    for top_bigram, measure in bigrams_with_measure.items():
        if top_bigram in bare_bigrams_to_filter:
            topk_bigrams_by_measure[top_bigram] = measure
            if len(topk_bigrams_by_measure) == topk:
                break

    return topk_bigrams_by_measure


def EstraiBigrammiPos(
    pos_tagged_tokens: List[Tuple[str, str]],
    wanted_POS_first: List[str],
    wanted_POS_second: List[str],
) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    POS_IDX = 1

    bigrammiEstratti = []
    bigrammiTokPos = nltk.bigrams(pos_tagged_tokens)
    for bigramma in bigrammiTokPos:
        if (
            bigramma[0][POS_IDX] in wanted_POS_first
            and bigramma[1][POS_IDX] in wanted_POS_second
        ):
            bigrammiEstratti.append(bigramma)

    return bigrammiEstratti


def filterBigramsByTokenFreq(
    bigrammiTokPos: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    distribFreqTokens: Dict[str, int],
    min_freq: int,
) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:

    filteredBigrams = []
    for bigrammaTokPos in bigrammiTokPos:

        allTokensHaveEnounghFreq = True
        for tokPos in bigrammaTokPos:
            tok = tokPos[0]
            if distribFreqTokens[tok] < min_freq:
                allTokensHaveEnounghFreq = False
        if allTokensHaveEnounghFreq:
            filteredBigrams.append(bigrammaTokPos)

    return filteredBigrams


def get_tokens_filterd_by_POS(
    pos_tagged_tokens: List[Tuple[str, str]], wanted_POS: List[str]
) -> List[str]:

    TOK_IDX = 0
    POS_IDX = 1

    POS_list: List[str] = []
    for pos_tagged_token in pos_tagged_tokens:
        if pos_tagged_token[POS_IDX] in wanted_POS:
            POS_list.append(pos_tagged_token[TOK_IDX])

    return POS_list


def filter_NE(
    NEs: Dict[Tuple[str, str], List[str]],
    wanted_POSes: List[str] = PROPER_NOUNS,
    wanted_NE_classes: List[str] = [PERSON_NE_CLASS],
) -> List[str]:

    named_entity_list = []
    for wanted_NE_class in wanted_NE_classes:
        for wanted_POS in wanted_POSes:
            key: Tuple[str, str] = (wanted_NE_class, wanted_POS)
            if key in NEs:
                named_entity_list += NEs[key]

    return named_entity_list


def get_all_NEs(
    tokensPOS: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], List[str]]:

    # individuato e classificato le Entità Nominate (NE) presenti nel testo

    NE_by_class_and_POS: Dict[Tuple[str, str], List[str]] = dict()
    ne_chunk = nltk.ne_chunk(tokensPOS)
    TOK_IDX = 0
    POS_IDX = 1
    for node in ne_chunk:
        is_intermediate_node = hasattr(node, "label")
        if is_intermediate_node:
            if node.label() in ["PERSON", "GPE", "ORGANIZATION"]:
                for leaf in node.leaves():
                    key: Tuple[str, str] = (node.label(), leaf[POS_IDX])
                    if key not in NE_by_class_and_POS:
                        NE_by_class_and_POS[key] = []
                    NE_by_class_and_POS[key].append(leaf[TOK_IDX])

    return NE_by_class_and_POS


def filter_sentences(
    sentences: List[str],
    words_and_freqs: Dict[str, int],
    min_lenght: int,
    max_lenght: int,
    min_token_freq: int,
):
    filtered_sentences = []
    for sentence in sentences:
        sentence_tokens = nltk.word_tokenize(sentence)
        sentence_lenght = len(sentence_tokens)
        if min_lenght <= sentence_lenght <= max_lenght:
            all_tokens_frequent_enough = True
            for token in sentence_tokens:
                if words_and_freqs[token] < min_token_freq:
                    all_tokens_frequent_enough = False
            if all_tokens_frequent_enough:
                filtered_sentences.append(sentence)

    return filtered_sentences


def get_sentences_with_markov2_probs(
    sentences: List[str],
    word_and_freqs: Dict[str, int],
    tokens_count: int,
    bigrams_with_conditioned_probability,
    trigrams_with_conditioned_probability,
):

    sentences_with_markov2_probs: Dict[str, float] = dict()
    for sentence in sentences:
        sentence_tokens = nltk.word_tokenize(sentence)
        sentence_prob = get_sentence_prob_markov2(
            sentence_tokens,
            word_and_freqs,
            tokens_count,
            bigrams_with_conditioned_probability,
            trigrams_with_conditioned_probability,
        )
        sentences_with_markov2_probs[sentence] = sentence_prob

    return SortDecreasing(sentences_with_markov2_probs)


def get_sentences_with_average_token_freq(
    sentences: List[str],
    word_and_freqs: Dict[str, int],
) -> Dict[str, float]:
    """

    :param sentences:
    :param word_and_freqs:
    :return: a dictionary with key the sentence text, and value the average token frequency.
    The dictionary is sorted in decreasing order by values.
    """
    sentences_with_average_token_freq: Dict[str, float] = dict()

    for sentence in sentences:
        sentence_tokens = nltk.word_tokenize(sentence)

        token_freqs_sum = 0
        for token in sentence_tokens:
            token_freqs_sum += word_and_freqs[token]
        average_token_freq = token_freqs_sum / len(sentence_tokens)
        sentences_with_average_token_freq[sentence] = average_token_freq

    return SortDecreasing(sentences_with_average_token_freq)


def get_sentence_prob_markov2(
    sentence_tokens,
    word_and_freqs,
    tokens_count: int,
    bigrams_with_conditioned_probability,
    trigrams_with_conditioned_probability,
):
    sentence_prob = 1.0

    first_token: str = sentence_tokens[0]
    first_token_prob = word_and_freqs[first_token] / tokens_count
    sentence_prob *= first_token_prob
    if len(sentence_tokens) == 1:
        return sentence_prob

    first_bigram = (sentence_tokens[0], sentence_tokens[1])
    first_bigram_prob = bigrams_with_conditioned_probability[first_bigram]
    sentence_prob *= first_bigram_prob
    if len(sentence_tokens) == 2:
        return sentence_prob

    sentence_trigrams = nltk.trigrams(sentence_tokens)
    for trigram in sentence_trigrams:
        trigram_prob = trigrams_with_conditioned_probability[trigram]
        sentence_prob *= trigram_prob

    return sentence_prob


def getFileAnalisysInfo(filepath: str) -> Dict:
    with open(filepath, mode="r", encoding="utf-8") as fileInput:
        raw = fileInput.read()

    #  il numero di frasi e di token:
    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    frasi: List[str] = sent_tokenizer.tokenize(raw)
    tokensTOT, pos_tagged_tokens = GetPosTaggedTokens(frasi)

    file_analisys_info: Dict[str, Any] = dict()
    file_analisys_info["filename"] = os.path.basename(filepath)
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

    # TODO:
    # #  distribuzione in termini di percentuale dell’insieme delle parole piene (Aggettivi, Sostantivi,
    #     # Verbi, Avverbi) e delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi).


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
    # TODO
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


def SortDecreasing(sort_me: Dict) -> Dict:
    return dict(sorted(sort_me.items(), key=lambda x: x[1], reverse=True))


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

    # TODO:
    # #  distribuzione in termini di percentuale dell’insieme delle parole piene (Aggettivi, Sostantivi,
    #     # Verbi, Avverbi) e delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi).


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
