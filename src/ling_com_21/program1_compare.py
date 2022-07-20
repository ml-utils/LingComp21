import os
import sys
from typing import Dict, Any

from utils import (
    EstraiSequenzaPos,
    count_avg_token_lenght,
    get_hapax_count,
    get_incremental_vocab_info,
    get_percentage_of_word_classes,
    CONTENT_WORDS,
    FUNCTIONAL_WORDS,
    analize_files_and_print_results, get_basic_file_info,
)


def program1_compare(
    filepath: str,
) -> Dict[str, Any]:

    file_analisys_info: Dict[str, Any] = dict()
    file_analisys_info["filename"] = os.path.basename(filepath)
    frasi, tokensTOT, pos_tagged_tokens = get_basic_file_info(filepath)

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

    file_analisys_info["num_hapax_first_1000_tokens"] = get_hapax_count(
        tokensTOT, tokens_limit=1000
    )
    file_analisys_info["incremental_vocab_info"] = get_incremental_vocab_info(
        tokensTOT, corpus_lenght_increments=500
    )

    file_analisys_info["perc_content_words"] = get_percentage_of_word_classes(
        pos_tagged_tokens, CONTENT_WORDS
    )
    file_analisys_info["perc_functional_words"] = get_percentage_of_word_classes(
        pos_tagged_tokens, FUNCTIONAL_WORDS
    )

    return file_analisys_info

def print_results_helper_pt1(file_analisys_info1, file_analisys_info2):

    # TODO: output: file di testo contenenti l'output ben formattato dei programmi.

    filename1 = file_analisys_info1["filename"]
    filename2 = file_analisys_info2["filename"]

    # TODO convert prints in writing to file
    # with open(filepath, mode="r", encoding="utf-8") as fileInput:
    #     raw = fileInput.read()

    print(f"Analisi dei due testi {filename1} e {filename2} :")

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

    print(
        f"La grandezza del vocabolario e la ricchezza lessicale (Type Token Ratio, TTR):"
    )
    for file_analisys_info in [file_analisys_info1, file_analisys_info2]:
        print(f"{filename1} :")
        for corpus_limit in file_analisys_info["incremental_vocab_info"]:
            vocab_size = file_analisys_info["incremental_vocab_info"][corpus_limit][
                "vocab_size"
            ]
            TTR = file_analisys_info["incremental_vocab_info"][corpus_limit]["TTR"]
            print(
                f"Corpus lenght: {corpus_limit}, vocab_size: {vocab_size}, TTR: {TTR}"
            )

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
        extraction_function=program1_compare,
        output_function=print_results_helper_pt1,
    )


if __name__ == "__main__":
    main()
