# Lexical chains are sequences of related words in a text that are connected through their meanings
# Connections are often based on Synonyms (eg car and automobile), Hypernyms (eg dog is a hyponom of animal) and Meronyms (eg wheel is a part of car)
# Hypernyms are words that represent a general category or class under which more specific words (hyponyms) fall
# Meronyms are words that refer to parts or components of a whole. They represent the relationship where one word denotes a part of something else.

import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Download required NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("averaged_perceptron_tagger")

def build_lexical_chains(sentence):
    """
    Builds lexical chains from a given sentence.

    Args:
        sentence (str): Input sentence.
    
    Returns:
        dict: Lexical chains where keys are chain indices and values are word lists.
    """
    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    # POS tagging for refined filtering
    pos_tags = pos_tag(filtered_tokens)

    # Filter out non-content words based on POS tags (e.g., prepositions, conjunctions)
    content_tokens = [word for word, pos in pos_tags if pos.startswith("NN") or pos.startswith("VB")]

    # Identify named entities and merge them as single units
    named_entities = ne_chunk(pos_tags)
    processed_tokens = []
    for subtree in named_entities:
        if hasattr(subtree, "label"):  # If it's a named entity
            processed_tokens.append(" ".join([leaf[0] for leaf in subtree]))
        else:
            word, pos = subtree
            if word in content_tokens:
                processed_tokens.append(word)

    # Create an empty list to hold the chains
    chains = []

    # Function to check if a word is semantically related to any word in a chain
    def is_related(word, chain):
        for chain_word in chain:
            if are_words_related(word, chain_word):
                return True
        return False

    # Function to check semantic relationship using WordNet
    def are_words_related(word1, word2):
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        if not synsets1 or not synsets2:
            return False
        # Use the path similarity measure
        for syn1 in synsets1:
            for syn2 in synsets2:
                similarity = syn1.path_similarity(syn2)
                if similarity and similarity > 0.4:  # Stricter threshold for similarity
                    return True
        return False

    # Build chains
    for word in processed_tokens:
        added_to_chain = False
        for chain in chains:
            if is_related(word, chain):
                chain.append(word)
                added_to_chain = True
                break
        if not added_to_chain:
            chains.append([word])

    # Convert chains to a dictionary for readability
    chain_dict = {i + 1: chain for i, chain in enumerate(chains)}
    return chain_dict

# Input sentence from the user
sentence = input("Enter a sentence: ")

# Build lexical chains
lexical_chains = build_lexical_chains(sentence)

# Display the lexical chains
print("\nLexical Chains:")
for chain_id, chain in lexical_chains.items():
    print(f"Chain {chain_id}: {chain}")
