# Chunking is a process of extracting meaningful phrases or segemnts from text based on syntactic rules. 
# It groups words into more informative phrases like noun phrase, verb phrase, prepositional phrase using POS tagging.
# POS is part of speech tagging like classifying words into nouns, verbs, prepositions etc.
# Example The curious cat sat on the warm mat. Here NP the curious cat, VP sat on the warm mat, PP on the warm mat

import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def chunk_sentence(sentence):
    """
    Performs chunking on a given sentence and displays extracted phrases.

    Args:
        sentence (str): Input sentence in English.
    """
    # Step 1: Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Step 2: Perform part-of-speech (POS) tagging
    pos_tags = pos_tag(words)
    

    # Step 3: Define the chunking grammar
    #DT is determiner, JJ  is adjectives, NN is noun, IN is preposition
    # ? optional, * 0 or more times, | or, + 1 or more times
    grammar = r"""
    NP: {<DT>?<JJ>*<NN>}   # Noun Phrase 
    VP: {<VB.*><NP|PP>*}   # Verb Phrase
    PP: {<IN><NP>}         # Prepositional Phrase
    """

    # Step 4: Create a chunk parser
    chunk_parser = RegexpParser(grammar)

    # Step 5: Perform chunking
    chunk_tree = chunk_parser.parse(pos_tags)
    print("\nChunk Tree:")
    print(chunk_tree)

    # Step 6: Extract and display chunks
    for subtree in chunk_tree.subtrees():
        if subtree.label() in ["NP", "VP", "PP"]:
            phrase = " ".join(word for word, tag in subtree.leaves())
            print(f"{subtree.label()}: {phrase}")

    # Optional: Display the chunk tree
    chunk_tree.draw()

# Input sentence from the user
sentence = input("Enter a sentence: ")

# Perform chunking
chunk_sentence(sentence)
