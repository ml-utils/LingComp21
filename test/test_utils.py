from unittest import TestCase

from src.ling_com_21.utils import (
    get_sentence_prob_markov2,
    get_bigrams_with_frequency,
    get_bigrams_with_measures,
    get_trigrams_with_frequency,
    get_trigrams_with_conditioned_probability,
)


class TestUtils(TestCase):

    # todo: all methods that calculate probabilities or frequency, check for missing
    # words, bigrams, or trigrams, and return 0.
    # test test_get_sentence_prob_markov2 with missing word, bigram, or trigram

    def test_get_sentence_prob_markov2(self):
        sentence_0_tokens = []
        sentence_1_tokens = ["cat"]
        sentence_with_new_bigram_tokens = ["cat", "."]
        sentence_2 = "The cat is on the table."
        sentence_2_tokens = ["The", "cat", "is", "on", "the", "table", "."]
        word_and_freqs = {
            "The": 1,
            "cat": 1,
            "is": 1,
            "on": 1,
            "the": 1,
            "table": 1,
            ".": 1,
        }
        tokens_count = len(word_and_freqs)
        tokensTOT = sentence_2_tokens
        bigrams_with_frequency = get_bigrams_with_frequency(tokensTOT)
        trigrams_with_frequency = get_trigrams_with_frequency(tokensTOT)
        _, bigrams_with_conditioned_probability = get_bigrams_with_measures(
            tokensTOT,
            word_and_freqs,
            bigrams_with_frequency,
        )
        trigrams_with_conditioned_probability = (
            get_trigrams_with_conditioned_probability(
                tokensTOT,
                bigrams_with_frequency,
                trigrams_with_frequency,
            )
        )

        sentences = [
            sentence_0_tokens,
            sentence_1_tokens,
            sentence_with_new_bigram_tokens,
        ]
        expected_probs = [0, 1 / len(tokensTOT), 0]
        for sentence_tokens, expected_prob in zip(sentences, expected_probs):
            actual_prob = get_sentence_prob_markov2(
                sentence_tokens,
                word_and_freqs,
                tokens_count,
                bigrams_with_conditioned_probability,
                trigrams_with_conditioned_probability,
            )
            error_msg = f"actual_prob={actual_prob}, expected_prob={expected_prob}, sentence_tokens={sentence_tokens}"
            assert actual_prob == expected_prob, error_msg
