#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot

#problem("unicode1", "Understanding Unicode", "1 point")[
#prompt-box[
#section-label[Prompt]

(a) What Unicode character does chr(0) return?

#deliverable[A one-sentence response.]

(b) How does this character’s string representation (\_\_repr\_\_()) differ from its printed
representation?

#deliverable[A one-sentence response.]

(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

#raw(">>> chr(0)\n>>> print(chr(0))\n>>> \"this is a test\" + chr(0) + \"string\"\n>>> print(\"this is a test\" + chr(0) + \"string\")", block: true, lang: "python")

#deliverable[A one-sentence response.]

]

#answer-box[
#section-label[Answer]

a. 是编号为0的一个字符NUL，解释器会把这个转义到'\\x00'展示出来。

b. repr 会把不可见或需要转义的字符用转义写法表示出来。print的时候把实际包含的字符显示出来。而\\x00没有可见字形，所以看起来像没有显示。

c. "this is a test" + chr(0) + "string" 是转义后的形式，print的话就是输出字符串里实际包含的字符，chr(0)这个字符仍然在字符串里，只是打印时不可见。

#raw(">>> chr(0)\n>>> print(chr(0))\n>>> \"this is a test\" + chr(0) + \"string\"\n>>> print(\"this is a test\" + chr(0) + \"string\")", block: true)
]
]
