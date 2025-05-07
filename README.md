# MLXIncon


## Datasets
Datasets consist of n knowledge bases(KBs), and the corresponding inconsistency degree value.
Each knowledge base is a string of the contained formulas, where each formula is separated by a whitespace.
E.g., for the MI-measure:
- ("a !a", 1.0)

Importantly: If the knowledge base is consistent, an additional element "consistent" is added. This can be accessed by the learning algorithm.

The knowledge bases are randomly created with the following parameters:
- (n - number of knowlegde bases)
- max number of atoms to be considered
- max number of formulas for each knowledge base

Each formula can have one of the following forms (at random):
- A simple literal, e.g., "a" or the negated form "!a"
- AND, i.e., "L1&&L2", where L1, L2 are simple literals
- OR, i.e., "L1||L2", where L1, L2 are simple literals

(Note that for AND and OR, only one connector is considered (and we don't have something like "a&&b&&c"))
(Note also that for AND and OR, the literals are sorted alphabetically to reduce the number of distinct representations, i.e., "a&&b" and "b&&a" are both stored as "a&&b")

The files of the names have the following convention:
"{Number}-kbs__{measure}__max-{n}-atoms__max-{5}-elements.txt"

For example, "10T-kbs__MI-measure__max-3-atoms__max-5-elements.txt" means:
- there are 10 thousand KBs, each of the form like ("a !a", 1.0)
- the measure in this case is the MI-measure
- max. 3 atoms are considered (a,b,c)
- every KB has 1-max.5 formulas (see above for forms)


