import os
import sys
from typing import Dict, Any

from utils import (
    CONTENT_WORDS,
    FUNCTIONAL_WORDS,
    EstraiSequenzaPos,
    count_avg_token_lenght,
    get_hapax_count,
    get_incremental_vocab_info,
    get_percentage_of_word_classes,
    analize_files_and_print_results,
    get_basic_file_info,
)


def extract_info1(
    filepath: str,
) -> Dict[str, Any]:
    """
    Dato in input il percorso di un file di testo contenente un corpus,
    ne estrae le seguenti informazioni statistiche, salvandole in un dizionario:
      il numero di frasi e di token;
      la lunghezza media delle frasi in termini di token e dei token (escludendo la punteggiatura)
     in termini di caratteri;
      il numero di hapax sui primi 1000 token;
      la grandezza del vocabolario e la ricchezza lessicale calcolata attraverso la Type Token
     Ratio (TTR), in entrambi i casi calcolati all'aumentare del corpus per porzioni incrementali
     di 500 token;
      distribuzione in termini di percentuale dell’insieme delle parole piene (Aggettivi, Sostantivi,
     Verbi, Avverbi) e delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi).
    :param filepath:
    :return:
    """
    file_analisys_info: Dict[str, Any] = dict()
    file_analisys_info["filename"] = os.path.basename(filepath)
    frasi, tokensTOT, pos_tagged_tokens = get_basic_file_info(filepath)

    # Estrae il numero di frasi e di token
    file_analisys_info["num_sentences"] = len(frasi)
    file_analisys_info["num_tokens"] = len(tokensTOT)

    # Estrae il numero medio di token in una frase (escludendo la punteggiatura)
    listaPOS_SenzaPunteggiatura = EstraiSequenzaPos(
        pos_tagged_tokens, exclude_punctuation=True
    )
    file_analisys_info["avg_tokens_per_sentence"] = len(
        listaPOS_SenzaPunteggiatura
    ) / len(frasi)

    # Estrae numero medio di caratteri per token
    tokens_count, charsTOTcount, avg_chars_per_token = count_avg_token_lenght(
        pos_tagged_tokens, exclude_punctuation=True
    )
    file_analisys_info["avg_chars_per_token"] = avg_chars_per_token

    # Estrae il numero di hapax sui primi 1000 token;
    file_analisys_info["num_hapax_first_1000_tokens"] = get_hapax_count(
        tokensTOT, tokens_limit=1000
    )

    # Estrae la grandezza del vocabolario e la ricchezza lessicale calcolata attraverso la Type Token
    #  Ratio (TTR), in entrambi i casi calcolati all'aumentare del corpus per porzioni incrementali
    # di 500 token;
    file_analisys_info["incremental_vocab_info"] = get_incremental_vocab_info(
        tokensTOT, corpus_lenght_increments=500
    )

    # Estrae la distribuzione in termini di percentuale
    # dell’insieme delle parole piene (Aggettivi, Sostantivi, Verbi, Avverbi)
    # e delle parole funzionali (Articoli, Preposizioni, Congiunzioni, Pronomi).
    file_analisys_info["perc_content_words"] = get_percentage_of_word_classes(
        pos_tagged_tokens, CONTENT_WORDS
    )
    file_analisys_info["perc_functional_words"] = get_percentage_of_word_classes(
        pos_tagged_tokens, FUNCTIONAL_WORDS
    )

    return file_analisys_info


def save_well_formatted_output_to_text_file(file_analisys_info1, file_analisys_info2):
    """
    Salva su file output di testo, le informazioni ben formattate
    contenute nei due dizionari in input (corrispondenti ognuno ad un corpus
    precedentemente analizzato).
    :param file_analisys_info1:
    :param file_analisys_info2:
    :return:
    """
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
    """
    Si aspetta come argomenti da linea di comando i path di due file di testo da analizzare.
    Estrae informazioni dai due file, e salva in output il confronto ben formattato tra i due.
    :return:
    """
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
        output_function=save_well_formatted_output_to_text_file,
    )


if __name__ == "__main__":
    main()
