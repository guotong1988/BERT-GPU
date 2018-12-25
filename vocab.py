from nltk.tokenize import wordpunct_tokenize
f = open("sample_text.txt",mode="r",encoding="utf-8")
f2 = open("vocab.txt",mode="w",encoding="utf-8")
lines = f.readlines()
vocab = set()
for line in lines:
    word_list = wordpunct_tokenize(line)
    for word in word_list:
        vocab.add(word)

f2.write("[PAD]\n")
f2.write("[UNK]\n")
f2.write("[CLS]\n")
f2.write("[SEP]\n")
f2.write("[MASK]\n")
for word in list(vocab):
    f2.write(word)
    f2.write("\n")

f2.close()
f.close()