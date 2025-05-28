from lightllm.server.router.model_infer.mode_backend.chunked_prefill.pre3_core.core import NT, T

numbers = {i: T(str(i)) for i in range(10)}
alphabet = {x: T(x) for x in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
symbols = {x: T(x) for x in "~`!@#$%^&*()_+-=[]{}|;:,.<>/?'ĊĠ"}
space = T(" ")
backstash = T(chr(92))
double_quote = T('"')

num_grammar = [
    (NT("NUMBER"), [NT("NUM")]),
    (NT("NUM"), [NT("INT")]),
    (NT("NUM"), [NT("FLOAT")]),
    (NT("INT"), [NT("DIGIT"), NT("INT")]),
    (NT("INT"), [NT("DIGIT")]),
    (NT("FLOAT"), [NT("INT"), symbols["."], NT("INT")]),
] + [(NT("DIGIT"), [x]) for x in numbers.values()]

# LR(1) expression grammar
# 1. E  -> E + T
# 2. E  -> E - T
# 3. E  -> T
# 4. T  -> T * F
# 5. T  -> T / F
# 6. T  -> F
# 7. F  -> ( E )
# 8. F  -> NUM
# 9. NUM -> DIGIT+
# 10. NUM -> DIGIT+ . DIGIT+
# 11. DIGIT -> [0-9]
expr_grammar = [
    (NT("EXPR"), [NT("E")]),
    (NT("E"), [NT("E"), symbols["+"], NT("T")]),
    (NT("E"), [NT("E"), symbols["-"], NT("T")]),
    (NT("E"), [NT("T")]),
    (NT("T"), [NT("T"), symbols["*"], NT("F")]),
    (NT("T"), [NT("T"), symbols["/"], NT("F")]),
    (NT("T"), [NT("F")]),
    (NT("F"), [symbols["("], NT("E"), symbols[")"]]),
    (NT("F"), [NT("NUMBER")]),
] + num_grammar

char_grammar = (
    [(NT("CHAR"), [x]) for x in alphabet.values()]
    + [(NT("CHAR"), [x]) for x in symbols.values()]
    + [(NT("CHAR"), [x]) for x in numbers.values()]
    + [(NT("CHAR"), [space])]
)

# LR(1) string grammar
# 1. STRING -> " CHARS "
# 2. CHARS -> CHAR CHARS
# 3. CHARS -> ESCAPE CHARS
# 4. CHARS -> ε
# 5. CHAR -> [^"\\]  // 任何非引号、非反斜杠字符
# 6. ESCAPE -> \"
# 7. ESCAPE -> \\
# 8. ESCAPE -> \/
# 9. ESCAPE -> \b
# 10. ESCAPE -> \f
# 11. ESCAPE -> \n
# 12. ESCAPE -> \r
# 13. ESCAPE -> \t
# 14. ESCAPE -> \u HEX HEX HEX HEX
# 15. HEX -> [0-9] | [a-f] | [A-F]
string_grammar = (
    [
        (NT("STRING"), [NT("STR")]),
        (NT("STR"), [double_quote, NT("CHARS"), double_quote]),
        (NT("STR"), [double_quote, double_quote]),
        (NT("CHARS"), [NT("CHAR"), NT("CHARS")]),
        (NT("CHARS"), [NT("ESCAPE"), NT("CHARS")]),
        (NT("CHARS"), [NT("CHAR")]),
        (NT("CHARS"), [NT("ESCAPE")]),
    ]
    + char_grammar
    + [
        (NT("ESCAPE"), [backstash, double_quote]),
        (NT("ESCAPE"), [backstash, backstash]),
        (NT("ESCAPE"), [backstash, symbols["/"]]),
        (NT("ESCAPE"), [backstash, alphabet["b"]]),
        (NT("ESCAPE"), [backstash, alphabet["f"]]),
        (NT("ESCAPE"), [backstash, alphabet["n"]]),
        (NT("ESCAPE"), [backstash, alphabet["r"]]),
        (NT("ESCAPE"), [backstash, alphabet["t"]]),
        (NT("ESCAPE"), [backstash, alphabet["u"], NT("HEX"), NT("HEX"), NT("HEX"), NT("HEX")]),
        (NT("HEX"), [NT("DIGIT")]),
        (NT("HEX"), [alphabet["a"]]),
        (NT("HEX"), [alphabet["b"]]),
        (NT("HEX"), [alphabet["c"]]),
        (NT("HEX"), [alphabet["d"]]),
        (NT("HEX"), [alphabet["e"]]),
        (NT("HEX"), [alphabet["f"]]),
        (NT("HEX"), [alphabet["A"]]),
        (NT("HEX"), [alphabet["B"]]),
        (NT("HEX"), [alphabet["C"]]),
        (NT("HEX"), [alphabet["D"]]),
        (NT("HEX"), [alphabet["E"]]),
        (NT("HEX"), [alphabet["F"]]),
    ]
)

# LR(1) JSON grammar
# 1. JSON -> Object
# 2. JSON -> Array

# 3. Object -> { Members }
# 4. Object -> { }
# 5. Members -> Pair
# 6. Members -> Pair , Members
# 7. Pair -> STRING : Value

# 8. Array -> [ Elements ]
# 9. Array -> [ ]
# 10. Elements -> Value
# 11. Elements -> Value , Elements

# 12. Value -> STRING
# 13. Value -> NUMBER
# 14. Value -> Object
# 15. Value -> Array
# 16. Value -> true
# 17. Value -> false
# 18. Value -> null
json_grammar = (
    [
        (NT("JSON"), [NT("JS")]),
        (NT("JS"), [NT("Object")]),
        (NT("JS"), [NT("Array")]),
        (NT("Object"), [symbols["{"], NT("Members"), symbols["}"]]),
        (NT("Object"), [symbols["{"], symbols["}"]]),
        (NT("Members"), [NT("Pair")]),
        (NT("Members"), [NT("Pair"), symbols[","], NT("Members")]),
        (NT("Pair"), [NT("STRING"), symbols[":"], NT("Value")]),
        (NT("Array"), [symbols["["], NT("Elements"), symbols["]"]]),
        (NT("Array"), [symbols["["], symbols["]"]]),
        (NT("Elements"), [NT("Value")]),
        (NT("Elements"), [NT("Value"), symbols[","], NT("Elements")]),
        (NT("Value"), [NT("STRING")]),
        (NT("Value"), [NT("NUMBER")]),
        (NT("Value"), [NT("Object")]),
        (NT("Value"), [NT("Array")]),
        (NT("Value"), [alphabet["t"], alphabet["r"], alphabet["u"], alphabet["e"]]),
        (NT("Value"), [alphabet["f"], alphabet["a"], alphabet["l"], alphabet["s"], alphabet["e"]]),
        (NT("Value"), [alphabet["n"], alphabet["u"], alphabet["l"], alphabet["l"]]),
    ]
    + string_grammar
    + num_grammar
)

"""
value: dict
         | list
         | ESCAPED_STRING
         | signed_number
         | "t" "r" "u" "e" | "f" "a" "l" "s" "e" | "n" "u" "l" "l"
list : "[" [value ("," value)*] "]"
dict : "{" [pair ("," pair)*] "}"
pair : ESCAPED_STRING ":" value

digit: "0".."9"
hexdigit: "a".."f"|"A".."F"|digit

int: digit+
signed_int: ["+"|"-"] int
decimal: int "." int? | "." int

_exp: ("e"|"E") signed_int
float: int _exp | decimal _exp?
signed_float: ["+"|"-"] float

number: float | int
signed_number: ["+"|"-"] number
"""
