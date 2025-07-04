import re


def extract_innermost_paren_content(expr):
    stack = []
    content = ""
    max_depth = 0
    current_depth = 0
    start_idx = -1

    for idx, char in enumerate(expr):
        if char == '(':
            current_depth += 1
            stack.append(idx)
            if current_depth > max_depth:
                max_depth = current_depth
                start_idx = idx
        elif char == ')':
            if stack:
                current_depth -= 1
                if current_depth == max_depth - 1:
                    end_idx = idx
                    content = expr[start_idx + 1:end_idx]
                stack.pop()

    return content if content else expr


def is_single_char_const(term):
    return len(term) == 1 and term.isalpha()


def is_double_char_var(term):
    return len(term) == 2 and term.isalpha()


def trim_common_affixes(term1, term2):
    term1_list = list(term1)
    term2_list = list(term2)

    # Trim common prefixes
    while term1_list and term2_list and term1_list[0] == term2_list[0]:
        term1_list.pop(0)
        term2_list.pop(0)

    # Trim common suffixes
    while term1_list and term2_list and term1_list[-1] == term2_list[-1]:
        term1_list.pop()
        term2_list.pop()

    return ''.join(term1_list), ''.join(term2_list)


def substitute_var(term, old_var, new_var):
    pattern = r'\b{}\b'.format(re.escape(old_var))
    return re.sub(pattern, new_var, term)


def apply_substitutions(terms, substitutions):
    for var, replacement in substitutions.items():
        for i in range(len(terms)):
            terms[i] = substitute_var(terms[i], var, replacement)
    return terms


def normalize_substitutions(subs_dict):
    normalized = {}
    for key, value in subs_dict.items():
        sanitized_key = re.sub(r"[^\w\s]", "", key)
        sanitized_val = re.sub(r"[^\w\s]", "", value)

        key_inner = extract_innermost_paren_content(sanitized_key)
        val_inner = extract_innermost_paren_content(sanitized_val)

        if is_single_char_const(key_inner) and is_double_char_var(val_inner):
            normalized[value] = key
        else:
            normalized[key] = value
    return normalized


def compute_mgu(left_args, right_args):
    substitutions = {}
    current_left = left_args.copy()
    current_right = right_args.copy()

    for idx in range(len(current_left)):
        arg1 = current_left[idx]
        arg2 = current_right[idx]

        if arg1 == arg2:
            continue

        inner1 = extract_innermost_paren_content(arg1)
        inner2 = extract_innermost_paren_content(arg2)

        if is_single_char_const(inner1) and is_single_char_const(inner2):
            if inner1 != inner2:
                return {}
            elif arg1 != arg2:
                return {}

        elif is_double_char_var(inner1) and is_double_char_var(inner2):
            if inner1 != inner2:
                stripped1, stripped2 = trim_common_affixes(arg1, arg2)
                substitutions[stripped2] = stripped1
                current_left = apply_substitutions(current_left, {stripped2: stripped1})
                current_right = apply_substitutions(current_right, {stripped2: stripped1})
            else:
                return {}

        elif is_double_char_var(inner1) and is_single_char_const(inner2):
            stripped1, stripped2 = trim_common_affixes(arg1, arg2)
            substitutions[stripped1] = stripped2
            current_left = apply_substitutions(current_left, {stripped1: stripped2})
            current_right = apply_substitutions(current_right, {stripped1: stripped2})

        else:
            stripped1, stripped2 = trim_common_affixes(arg1, arg2)
            substitutions[stripped2] = stripped1
            current_left = apply_substitutions(current_left, {stripped2: stripped1})
            current_right = apply_substitutions(current_right, {stripped2: stripped1})

    seen = set()
    for key, val in substitutions.items():
        if key in seen or val in seen:
            return {}
        seen.update({key, val})

    return substitutions


def parse_predicate(pred_str):
    name_part, rest = pred_str.split("(", 1)
    depth = 1
    end_idx = 0
    for idx, char in enumerate(rest):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                end_idx = idx
                break
    args_part = rest[:end_idx]
    args = [a.strip() for a in args_part.split(',')]
    return name_part.strip(), args


pred1 = input("请输入第一个原子公式: ").strip()
pred2 = input("请输入第二个原子公式: ").strip()

pname1, args1 = parse_predicate(pred1)
pname2, args2 = parse_predicate(pred2)

if pname1 != pname2:
    print("错误：谓词不同无法合一")
elif len(args1) != len(args2):
    print("错误：参数数量不同")
else:
    mgu = compute_mgu(args1, args2)
    if not mgu:
        print("MGU不存在")
    else:
        final_subs = normalize_substitutions(mgu)
        print("最广合一替换为:", final_subs)
