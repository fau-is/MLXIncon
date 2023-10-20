# MLXIncon


## Datasets
Datasets consist of n knowledge bases(KBs), and the corresponding inconsistency degree value.
Each knowledge base is a string of the contained formulas, where each formula is separated by a whitespace.
E.g., for the MI-measure:
- ("a !a", 1.0)

Importantly: If the knowledge base is consistent, an additional element "consistent" is added. This can be accessed by the learning algorithm.