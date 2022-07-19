# Penn Tree Bank tagset lists
ALL_PUNKTUATION = [".", ",", ":", "(", ")"]  # "SYM" (todo: verify what symbols include)
ADJECTIVES = ["JJ", "JJR", "JJS"]
ADVERBS = ["RB", "RBR", "RBS", "WRB"]
NOUNS = ["NN", "NNS", "NNP", "NNPS"]  # sostantivi
PROPER_NOUNS = ["NNP", "NNPS"]
VERBS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
CONTENT_WORDS = ADJECTIVES + NOUNS + VERBS + ADVERBS
ARTICLES = ["DT", "WDT"]
PREPOSITIONS = ["IN"]
CONJUNCTIONS = ["CC"]
OTHER_FUNCTIONAL = [
    "EX",  # Existential there
    "MD",  # Modal
    "PDT", # Predeterminer
    "POS", # possessive ending
    "RP", # particle
    "TO", # to
    "UH", # interjection
]
PRONOUNS = ["PRP", "PRP$", "WP", "WP$"]
FUNCTIONAL_WORDS = ARTICLES + PREPOSITIONS + CONJUNCTIONS + PRONOUNS + OTHER_FUNCTIONAL
OTHER = [
    "CD",  # CARDINAL NUMBER
    "FW",  # foreign word
    "LS",  # list item marker
    "SYM",  # symbol
]

# Named Entities classes
PERSON_NE_CLASS = "PERSON"  # , "GPE", "ORGANIZATION"