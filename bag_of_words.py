# This approach converts text data into numerical format, represents text as a set of words and their frequencies in it
# Position and order of words is ignored only thier occurence matters
# Steps tokenization then vocabulary creation and encoding

from collections import Counter
import re

def bag_of_words(sentence):
    """
    Converts a sentence into its Bag of Words representation.

    Args:
        sentence (str): Input sentence.
    
    Returns:
        dict: A dictionary representing the word counts (Bag of Words).
    """
    # Step 1: Preprocess the sentence
    # Convert to lowercase and remove punctuation
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)

    # Step 2: Tokenize the sentence into words
    words = sentence.split()

    # Step 3: Count the frequency of each word
    word_counts = Counter(words)

    return word_counts

# Input sentence from the user
sentence = input("Enter a sentence: ")

# Perform Bag of Words representation
bow = bag_of_words(sentence)

# Display the Bag of Words
print("\nBag of Words Representation:")
for word, count in bow.items():
    print(f"{word}: {count}")
