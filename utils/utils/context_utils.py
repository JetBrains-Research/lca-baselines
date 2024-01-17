import ast


def collect_good_context(row):
    context = ""
    row['good_code_files'] = ast.literal_eval(str(row['good_code_files']).replace('repos', '../data/repos_clean'))
    for con_path in row['good_code_files']:
        with open(con_path, 'r') as f:
            con = f.read()
            context += '\n\n' + con
    context = context.lstrip()
    return context


def trim_context(context, tokenizer, max_len):
    tokenized_context = tokenizer.encode(context, max_length=512_000, truncation=True)
    tokenized_context = tokenized_context[:max_len]
    detokenized_context = tokenizer.decode(tokenized_context)
    return detokenized_context