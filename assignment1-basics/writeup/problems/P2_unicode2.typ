#import "../template.typ": problem, deliverable, note, section-label, subhead, prompt-box, answer-box, plot

#problem("unicode2", "Unicode Encodings", "3 points")[
#prompt-box[
#section-label[Prompt]

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

#deliverable[A one-to-two sentence response.]

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

#raw("def decode_utf8_bytes_to_str_wrong(bytestring: bytes):\n    return \"\".join([bytes([b]).decode(\"utf-8\") for b in bytestring])\n\n>>> decode_utf8_bytes_to_str_wrong(\"hello\".encode(\"utf-8\"))\n'hello'", block: true, lang: "python")

#deliverable[An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation of why the function is incorrect.]

(c) Give a two-byte sequence that does not decode to any Unicode character(s).

#deliverable[An example, with a one-sentence explanation.]

]

#answer-box[
#section-label[Answer]

a. 因为对很多真实文本，UTF-16/UTF-32 会引入更多冗余字节，而UTF-8更省字节，训练 tokenizer 更省数据和计算。对 ASCII/英文尤其如此。

b. 这个函数把 UTF-8 按单个字节分别 decode，但很多 Unicode 字符需要多个字节共同表示。比如传入"牛".encode("utf-8") 时（即b'\\xe7\\x89\\x9b'），会报错。因为\\xe7是 leading byte，后面必须跟满足 10xxxxxx 模式的 continuation bytes，无法单独decode。

c. 比如说b'\\xe7\\xe7'。第一个 0xe7 是 1110xxxx，表示它开头的是一个 3-byte UTF-8 字符；但整个串只有 2 个字节，而且第二个 0xe7 也不是合法 continuation byte（不是 10xxxxxx）。
]
]
