# MLXIncon


## Datasets
Datasets consist of n knowledge bases(KBs), and the corresponding inconsistency degree values for the MI-measure and the AT-measure.
Each knowledge base is a string of the contained formulas, where each formula is separated by a whitespace.
E.g.:
- ("a !a a&&b !b", 3.0, 2.0)

In some cases, flags may be added (see heuristics in paper).

The knowledge bases are randomly created with the following parameters:
- (n - number of knowlegde bases)
- max number of atoms to be considered
- max number of formulas for each knowledge base

Each formula f can have one of the following forms (at random):
f:== L |  f1||f2  |   f1&&f2
, where L is a literal, i.e., an atom or its negated form. 

(Note that for AND and OR, the literals are sorted alphabetically to reduce the number of distinct representations, i.e., "a&&b" and "b&&a" are both stored as "a&&b")

The files of the names have the following convention:
"{Number}-kbs__max-{n}-atoms__max-{m}-elements.txt"

For example, "10T-kbs__max-3-atoms__max-5-elements.txt" means:
- there are 10 thousand KBs
- max. 3 atoms are considered (a,b,c)
- every KB has 1-max.5 formulas (see above for forms)


