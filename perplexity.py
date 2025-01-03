# Perplexity is a metric used to evaluate the quality of a probabilistic model
# It measures how well a language model predicts a sequence of words. Lower perplexity indicates that the model is better at predicting the test data.

import math

def calculate_perplexity(probabilities):
    """
    Calculates the perplexity of a language model given word probabilities.

    Args:
        probabilities (list of float): Probabilities of each word in the sequence.

    Returns:
        float: Perplexity score.
    """
    # Check for valid probabilities
    if not probabilities or any(p <= 0 or p > 1 for p in probabilities):
        raise ValueError("All probabilities must be in the range (0, 1].")

    # Calculate perplexity
    n = len(probabilities)
    log_sum = sum(math.log2(p) for p in probabilities)
    perplexity = 2 ** (-log_sum / n)

    return perplexity

# Example usage
# Sample probabilities from a language model for a word sequence
probabilities = [0.1, 0.2, 0.25, 0.4]  # These are example probabilities

# Calculate perplexity
try:
    perplexity = calculate_perplexity(probabilities)
    print(f"Perplexity: {perplexity:.4f}")
except ValueError as e:
    print(e)
