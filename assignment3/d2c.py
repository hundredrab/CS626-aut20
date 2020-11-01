import spacy
from nltk import Tree

nlp = spacy.load("en")
sent = "The boy is playing with a green-colored football"  # It's never too late to do something.'
sent = "My dog also likes eating sausages with gravy and also ham"
sent = "Here he goes"
sent = "Ram and Shyam"
sent = "Peter went to London"
# sent = "The cat is on the mat"
# sent = "The quick brown fox jumps over the lazy dog."
sent = "Time gains momentum"
sent = "Jack and Jill went up the hill"
doc = nlp(sent)
sent = list(doc.sents)[0]
# print(sent._.parse_string)


d = dict()


def p(node):
    return node.orth_ + node.tag_ + str(node.i) + node.pos_


def to_nltk_tree(node):
    if node.n_lefts or node.n_rights:
        return Tree(p(node), [to_nltk_tree(child) for child in node.children])
    else:
        d[node.i] = node.tag_
        return p(node)


it = 0

# sp = lambda x: tuple(sorted(x))
PRODUCTIONS = {
    ("NP", "VP"): "S",
    ("PRON"): "NP",
    ("PROPN"): "NP",
    ("DET", "Nominal"): "NP",
    ("Nominal", "NOUN"): "Nominal",
    ("NOUN"): "Nominal",
    ("VERB"): "VP",
    ("VERB", "NP"): "VP",
    ("VERB", "NP", "PP"): "VP",  # No use for this one
    ("VERB", "PP"): "VP",
    ("ADP", "NP"): "PP",

    ("ADP", "Nominal"): "NP"  # My rules
}

memo_chunk = dict()

def get_chunk_tag(node):
    if node.i in memo_chunk:
        return memo_chunk[node.i]
    if not (node.n_lefts or node.n_rights):
        if (node.pos_) in PRODUCTIONS:
            return PRODUCTIONS[node.pos_]
        return node.pos_
    curr = node.pos_
    num_children = node.n_lefts + node.n_rights
    lefts = list(node.lefts)[::-1]
    rights = list(node.rights)
    i = 0
    while num_children:
        if i > 10:
            print(f"ERR on {node}, with {lefts}, {rights}")
            return curr
        i += 1
        if node.i == 3:
            print(lefts, rights)
        if lefts:
            ll = get_chunk_tag(lefts[0]), curr
            if ll in PRODUCTIONS:
                num_children -= 1
                curr = PRODUCTIONS[ll]
                # print(curr, ':\t', list(lefts.pop(0).subtree), list(node.subtree))
                continue
        if rights:
            rr = curr, get_chunk_tag(rights[0])
            if rr in PRODUCTIONS:
                num_children -= 1
                curr = PRODUCTIONS[rr]
                # print(curr, ':\t', list(node.subtree) + list(rights.pop(0).subtree),)
                continue
        if curr in PRODUCTIONS:
            curr = PRODUCTIONS[node.pos_]
        # print("GOT NOTHING", node, lefts, rights)
        # print([get_chunk_tag(l) for l in lefts], node.pos_,  [get_chunk_tag(r) for r in rights])
        # print('-'*30)
    # print(curr, node, list(node.children))
    print(f"{curr}:\t{list(node.subtree)}")
    memo_chunk[node.i] = curr
    return curr


sss = [get_chunk_tag(s) for s in sent]
print("CONST TAG \t\t", "\t".join(sss))
print("SENT      \t\t", "\t".join([s.orth_ for s in sent]))
print("POS TAG   \t\t", "\t".join([s.pos_ for s in sent]))
# get_chunk_tag(sent.root)
# print(to_nltk_tree(sent.root))

print("\n\n")
(to_nltk_tree(sent.root))
# print(d)
