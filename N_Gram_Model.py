# N-Gram Model is a technique used in NLP to predict or analyse sequences of words in text, based on the likelihood of a word appearing after a sequence of N-1 words
# It assumes the probability of a word depends on previous N-1 words there are different types unigram(N=1),bigram (N=2), trigram (N=3)
# P(w1,w2,w3,w4...)=P(w1). P(w2|w1) . P(w3|w2)... P(wn|wn-1) kid of formula used if it is a bigram
import random
from collections import defaultdict, Counter

def generate_ngrams(tokens, n):
    """
    Generates n-grams from a list of tokens.
    
    Args:
        tokens (list): List of words/tokens.
        n (int): Order of the n-gram.
    
    Returns:
        list: List of n-grams as tuples.
    """
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def train_ngram_model(corpus, n):
    """
    Trains an n-gram model from a given corpus.
    
    Args:
        corpus (str): Input text corpus.
        n (int): Order of the n-gram.
    
    Returns:
        dict: N-Gram probabilities.
    """
    # Tokenize the corpus
    tokens = corpus.split()
    
    # Generate n-grams and (n-1)-grams
    ngrams = generate_ngrams(tokens, n)
    n_minus_one_grams = generate_ngrams(tokens, n - 1) if n > 1 else None
    
    # Count n-grams and (n-1)-grams
    ngram_counts = Counter(ngrams)
    n_minus_one_gram_counts = Counter(n_minus_one_grams) if n_minus_one_grams else None
    
    # Calculate probabilities
    ngram_probs = {}
    for ngram, count in ngram_counts.items():
        if n_minus_one_grams:
            prefix = ngram[:-1]
            ngram_probs[ngram] = count / n_minus_one_gram_counts[prefix]
        else:  # Unigram case
            ngram_probs[ngram] = count / len(tokens)
    
    return ngram_probs

def generate_sentence(ngram_probs, n, max_words=20):
    """
    Generates a sentence using the trained n-gram model.
    
    Args:
        ngram_probs (dict): N-Gram probabilities.
        n (int): Order of the n-gram.
        max_words (int): Maximum number of words in the generated sentence.
    
    Returns:
        str: Generated sentence.
    """
    # Start with a random n-gram
    ngrams = list(ngram_probs.keys())
    current_ngram = random.choice(ngrams)
    sentence = list(current_ngram)
    
    for _ in range(max_words - n):
        # Get possible next words
        candidates = [ngram for ngram in ngrams if ngram[:-1] == tuple(sentence[-(n - 1):])]
        if not candidates:
            break
        next_ngram = random.choice(candidates)
        sentence.append(next_ngram[-1])
    
    return ' '.join(sentence)

# Example Usage
if __name__ == "__main__":
    # Input corpus
    corpus = "I love natural language processing and I love coding in Python. Natural language processing is amazing."

    # Train a bigram model (n = 2)
    n = 2
    ngram_probs = train_ngram_model(corpus, n)

    # Generate a sentence
    generated_sentence = generate_sentence(ngram_probs, n)
    print("Generated Sentence:", generated_sentence)

    # Print N-Gram probabilities
    print("\nN-Gram Probabilities:")
    for ngram, prob in ngram_probs.items():
        print(f"{ngram}: {prob:.4f}")
