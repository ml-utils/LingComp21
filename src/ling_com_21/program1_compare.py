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
    analize_files_and_print_results,
    get_basic_file_info,
)


def extract_info1(
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

    filename1 = file_analisys_info1["filename"]
    filename2 = file_analisys_info2["filename"]

    output_filename = "output_prog1.txt"

    # TODO well formatted output
    with open(output_filename, mode="w", encoding="utf-8") as f:

        print(f"Analisi dei due testi {filename1} e {filename2} :\n", file=f)

        print(
            f"Numero di frasi: \n"
            f"{file_analisys_info1['num_sentences']} ({filename1}) e "
            f"{file_analisys_info2['num_sentences']} ({filename2}).",
            file=f,
        )

        print(
            f"Numero di token totali: \n"
            f"{file_analisys_info1['num_tokens']} ({filename1}) e "
            f"{file_analisys_info2['num_tokens']} ({filename2}).",
            file=f,
        )
        # I due testi hanno rispettivamente 364 e 567 frasi.
        # I due testi hanno rispettivamente 10339 e 6462 token totali.

        print(
            f"Numero medio di token in una frase (escludendo la punteggiatura): \n"
            f"{file_analisys_info1['avg_tokens_per_sentence']:.2f} ({filename1}) e "
            f"{file_analisys_info2['avg_tokens_per_sentence']:.2f} ({filename2}).",
            file=f,
        )

        print(
            f"Numero medio dei caratteri di un token (escludendo la punteggiatura): \n"
            f"{file_analisys_info1['avg_chars_per_token']:.2f} ({filename1}) e "
            f"{file_analisys_info2['avg_chars_per_token']:.2f} ({filename2}).",
            file=f,
        )

        print(
            f"Numero di hapax sui primi 1000 token: \n"
            f"{file_analisys_info1['num_hapax_first_1000_tokens']} ({filename1}) e "
            f"{file_analisys_info2['num_hapax_first_1000_tokens']} ({filename2}).",
            file=f,
        )

        print(
            f"\nLa grandezza del vocabolario (|V|) e la ricchezza lessicale (Type Token Ratio, TTR), \n"
            f"calcolati all'aumentare del corpus (|C|) per porzioni incrementali di 500 token: \n",
            file=f,
        )
        for file_analisys_info in [file_analisys_info1, file_analisys_info2]:
            print(f"{filename1} :", file=f)
            for corpus_limit in file_analisys_info["incremental_vocab_info"]:
                vocab_size = file_analisys_info["incremental_vocab_info"][corpus_limit][
                    "vocab_size"
                ]
                TTR = file_analisys_info["incremental_vocab_info"][corpus_limit]["TTR"]
                print(
                    f"Token: {corpus_limit}, vocabolario: {vocab_size}, TTR: {TTR}",
                    file=f,
                )

        print(
            f"Percentuale delle parole piene (Aggettivi, Sostantivi, Verbi, Avverbi) : \n"
            f"{file_analisys_info1['perc_content_words']:.2%} ({filename1}) e "
            f"{file_analisys_info2['perc_content_words']:.2%} ({filename2}).",
            file=f,
        )
        print(
            f"Percentuale delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi) : \n"
            f"{file_analisys_info1['perc_functional_words']:.2%} ({filename1}) e "
            f"{file_analisys_info2['perc_functional_words']:.2%} ({filename2}).",
            file=f,
        )

    print(f"written output to {output_filename}")


def main():
    if len(sys.argv) == 3:
        filepath1 = sys.argv[1]
        filepath2 = sys.argv[2]
    else:
        print(f"Errore, argomenti di input non corretti: {sys.argv}")
        # filepath1 = "..\\..\\Cablegate.txt"
        # filepath2 = "..\\..\\Colbert.txt"
        sys.exit(1)

    analize_files_and_print_results(
        filepath1=filepath1,
        filepath2=filepath2,
        extraction_function=extract_info1,
        output_function=print_results_helper_pt1,
    )


if __name__ == "__main__":
    main()
