class Token(str):
    def __new__(cls, type_, value):
        obj = str.__new__(cls, value)
        obj.type = type_
        obj.value = value
        return obj

    def __class_getitem__(cls, item):
        return cls

class Tree:
    def __init__(self, data, children=None):
        self.data = data
        self.children = children or []

    def __class_getitem__(cls, item):
        return cls

class Transformer:
    pass

class Lark:
    def __init__(self, grammar: str, parser: str = 'earley', maybe_placeholders: bool = False):
        pass

    def lex(self, text: str):
        import re
        pos = 0
        while pos < len(text):
            if text[pos] == ';':
                while pos < len(text) and text[pos] != '\n':
                    pos += 1
                continue
            if text[pos] in ' \t':
                start = pos
                while pos < len(text) and text[pos] in ' \t':
                    pos += 1
                yield Token('WS_INLINE', text[start:pos])
                continue
            if text[pos] == '\n':
                yield Token('NEWLINE', '\n')
                pos += 1
                continue
            if text[pos] == ',':
                yield Token('COMMA', ',')
                pos += 1
                continue
            if text[pos] == ':':
                yield Token('COLON', ':')
                pos += 1
                continue
            if text[pos] == '"':
                start = pos
                pos += 1
                while pos < len(text) and text[pos] != '"':
                    if text[pos] == '\\':
                        pos += 2
                    else:
                        pos += 1
                pos += 1
                yield Token('ESCAPED_STRING', text[start:pos])
                continue
            for kw in ['SECTION', 'defb', 'defw', 'defl', 'defs', 'defm']:
                if text.startswith(kw, pos):
                    yield Token(kw.upper(), kw)
                    pos += len(kw)
                    break
            else:
                m = re.match(r'(0x[0-9a-fA-F]+|[0-9a-fA-F]+[hH]|0b[01]+|[01]+[bB]|\d+)', text[pos:])
                if m:
                    value = m.group(0)
                    yield Token('NUMBER', value)
                    pos += len(value)
                    continue
                m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', text[pos:])
                if m:
                    value = m.group(0)
                    yield Token('CNAME', value)
                    pos += len(value)
                    continue
                pos += 1

    def parse(self, text: str):
        tokens = list(self.lex(text))
        idx = 0
        children = []
        while idx < len(tokens):
            if tokens[idx].type == 'NEWLINE':
                idx += 1
                continue
            line_tokens = []
            while idx < len(tokens) and tokens[idx].type != 'NEWLINE':
                line_tokens.append(tokens[idx])
                idx += 1
            if idx < len(tokens) and tokens[idx].type == 'NEWLINE':
                idx += 1
            line_children = []
            if len(line_tokens) >= 2 and line_tokens[1].type == 'COLON':
                label_tok = line_tokens.pop(0)
                line_tokens.pop(0)
                # drop leading whitespace after label
                while line_tokens and line_tokens[0].type == 'WS_INLINE':
                    line_tokens.pop(0)
                line_children.append(Tree('label', [label_tok]))
            if line_tokens:
                first = line_tokens[0]
                if first.type == 'SECTION':
                    line_children.append(Tree('section_decl', line_tokens))
                elif first.type in {'DEFB','DEFW','DEFL','DEFS','DEFM'}:
                    map_ = {
                        'DEFB':'defb_directive',
                        'DEFW':'defw_directive',
                        'DEFL':'defl_directive',
                        'DEFS':'defs_directive',
                        'DEFM':'defm_directive',
                    }
                    if first.type == 'DEFM' and line_tokens and line_tokens[-1].type == 'ESCAPED_STRING':
                        lit = line_tokens.pop()
                        # drop whitespace before string
                        if line_tokens and line_tokens[-1].type == 'WS_INLINE':
                            line_tokens.pop()
                        line_children.append(Tree(map_[first.type], [first, Tree('string_literal', [lit])]))
                    else:
                        line_children.append(Tree(map_[first.type], line_tokens))
                else:
                    line_children.append(Tree('instruction', line_tokens))
            if line_children:
                children.extend(line_children)
        return Tree('start', children)
