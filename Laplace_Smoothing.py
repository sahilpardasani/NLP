# Technique used in NLP and probability to handle the problem of zero probability. 
# In NLP if in the training data a word does not appear then it's probability becomes 0 so multiplying probabilities would give 0
# To avoid this we add a small value to all probailities
# P(w|context) = count(w, context) + 1 / total count of context + V 
from collections import Counter, defaultdict

def laplace_smoothing(bigrams, vocabulary_size):
    """
    Applies Laplace smoothing to bigram probabilities.

    Args:
        bigrams (list of tuples): List of bigram tuples.
        vocabulary_size (int): Total number of unique words in the vocabulary.

    Returns:
        dict: Smoothed probabilities for each bigram.
    """
    # Count frequencies of bigrams
    bigram_counts = Counter(bigrams)

    # Count frequencies of first words in bigrams
    unigram_counts = defaultdict(int)
    for (w1, w2) in bigrams:
        unigram_counts[w1] += 1

    # Compute smoothed probabilities
    smoothed_probs = {}
    for bigram, count in bigram_counts.items():
        w1 = bigram[0]
        smoothed_probs[bigram] = (count + 1) / (unigram_counts[w1] + vocabulary_size)

    return smoothed_probs


# Example usage
sentence = "I love NLP and I love coding"

# Tokenize the sentence
tokens = sentence.split()

# Create bigrams
bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

# Calculate vocabulary size (unique words)
vocabulary_size = len(set(tokens))

# Apply Laplace smoothing
smoothed_bigram_probs = laplace_smoothing(bigrams, vocabulary_size)

# Print smoothed probabilities
print("Smoothed Bigram Probabilities:")
for bigram, prob in smoothed_bigram_probs.items():
    print(f"{bigram}: {prob:.4f}")
