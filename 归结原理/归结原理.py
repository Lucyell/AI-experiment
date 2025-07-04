import re


def is_variable(symbol: str) -> bool:
    return '(' not in symbol and symbol[0] in 'xyzuvw'


def parse_literal(literal: str):
    lit = literal.strip()
    if lit.startswith('~'):
        sign = -1
        lit = lit[1:].strip()
    else:
        sign = +1

    if '(' not in lit:
        return (sign, lit, [])

    match = re.match(r'([A-Z][A-Za-z0-9_]*?)\((.*)\)$', lit)
    predicate_name = match.group(1)
    arguments_str = match.group(2).strip()
    arguments = []
    level = 0
    current_arg = []
    for char in arguments_str:
        if char == ',' and level == 0:
            arg = ''.join(current_arg).strip()
            if arg:
                arguments.append(arg)
            current_arg = []
        else:
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            current_arg.append(char)
    if current_arg:
        arg = ''.join(current_arg).strip()
        if arg:
            arguments.append(arg)

    return (sign, predicate_name, arguments)


def substitute(substitution: dict, literal: str) -> str:
    def replace_term(term: str):
        if is_variable(term) and term in substitution:
            return replace_term(substitution[term])
        if '(' in term:
            index = term.index('(')
            functor = term[:index]
            args_str = term[index + 1:-1]
            sub_args = []
            level = 0
            current_arg = []
            for char in args_str:
                if char == ',' and level == 0:
                    sub_args.append(''.join(current_arg).strip())
                    current_arg = []
                else:
                    if char == '(':
                        level += 1
                    elif char == ')':
                        level -= 1
                    current_arg.append(char)
            if current_arg:
                sub_args.append(''.join(current_arg).strip())

            new_sub_args = [replace_term(a) for a in sub_args]
            return f"{functor}({','.join(new_sub_args)})"
        return term

    (sign, predicate, arguments) = parse_literal(literal)
    new_arguments = [replace_term(a) for a in arguments]
    new_lit = f"{predicate}({','.join(new_arguments)})" if new_arguments else predicate
    if sign < 0:
        new_lit = '~' + new_lit
    return new_lit


def apply_substitution_to_clause(substitution: dict, clause: tuple) -> tuple:
    return tuple(sorted(substitute(substitution, lit) for lit in clause))


def occurs_check(variable: str, term: str, substitution: dict) -> bool:
    t_applied = substitute(substitution, term)
    if t_applied == variable:
        return True
    if '(' in t_applied:
        (s, p, args) = parse_literal(t_applied)
        for arg in args:
            if occurs_check(variable, arg, substitution):
                return True
    return False


def unify_terms(term1: str, term2: str, substitution: dict) -> dict or None:
    s_applied = substitute(substitution, term1)
    t_applied = substitute(substitution, term2)
    if s_applied == t_applied:
        return substitution

    if is_variable(s_applied):
        if occurs_check(s_applied, t_applied, substitution):
            return None
        new_subst = dict(substitution)
        new_subst[s_applied] = t_applied
        return new_subst

    if is_variable(t_applied):
        if occurs_check(t_applied, s_applied, substitution):
            return None
        new_subst = dict(substitution)
        new_subst[t_applied] = s_applied
        return new_subst

    (sign_s, functor_s, args_s) = parse_literal(s_applied)
    (sign_t, functor_t, args_t) = parse_literal(t_applied)
    if functor_s != functor_t or len(args_s) != len(args_t):
        return None
    new_subst = dict(substitution)
    for a_s, a_t in zip(args_s, args_t):
        new_subst = unify_terms(a_s, a_t, new_subst)
        if new_subst is None:
            return None
    return new_subst


def unify_literals(lit1: str, lit2: str) -> dict or None:
    (sign1, predicate1, args1) = parse_literal(lit1)
    (sign2, predicate2, args2) = parse_literal(lit2)

    if sign1 + sign2 != 0:
        return None
    if predicate1 != predicate2 or len(args1) != len(args2):
        return None

    substitution = {}
    for a1, a2 in zip(args1, args2):
        substitution = unify_terms(a1, a2, substitution)
        if substitution is None:
            return None
    return substitution


def resolution_first_order(knowledge_base: set[tuple]):
    initial_clauses = sorted(list(knowledge_base), key=lambda c: str(c))
    clauses_dict = {}
    next_id = 1
    result_lines = []

    for clause in initial_clauses:
        clauses_dict[next_id] = {
            'clause': clause,
            'parents': None
        }
        pretty = '(' + ','.join(clause) + ')'
        result_lines.append(f"{next_id} {pretty}")
        next_id += 1

    active_ids = list(clauses_dict.keys())

    while True:
        found_new_clause = False
        candidate_list = []
        for i1 in range(len(active_ids)):
            for i2 in range(i1 + 1, len(active_ids)):
                id1 = active_ids[i1]
                id2 = active_ids[i2]
                clause1 = clauses_dict[id1]['clause']
                clause2 = clauses_dict[id2]['clause']

                for idx1, lit1 in enumerate(clause1):
                    for idx2, lit2 in enumerate(clause2):
                        unifier = unify_literals(lit1, lit2)
                        if unifier is not None:
                            new_c1 = list(clause1[:idx1] + clause1[idx1 + 1:])
                            new_c2 = list(clause2[:idx2] + clause2[idx2 + 1:])
                            combined = new_c1 + new_c2
                            # Apply the unifier
                            new_clause = apply_substitution_to_clause(unifier, tuple(combined))
                            new_clause_set = set(new_clause)
                            new_clause = tuple(sorted(new_clause_set))
                            if not already_in_kb(new_clause, clauses_dict):
                                candidate_list.append(
                                    (id1, id2, idx1, idx2, new_clause, unifier)
                                )

        if candidate_list:
            candidate_list.sort(key=lambda tup: (len(tup[4]), tup[0], tup[1]))
            (parent1, parent2, index1, index2, new_clause, unifier) = candidate_list[0]

            clauses_dict[next_id] = {
                'clause': new_clause,
                'parents': ((parent1, index1), (parent2, index2))
            }
            active_ids.append(next_id)
            ref1 = format_reference(parent1, clauses_dict[parent1]['clause'], index1)
            ref2 = format_reference(parent2, clauses_dict[parent2]['clause'], index2)
            pretty_new = ','.join(new_clause)
            if unifier:
                result_uni = '{' + ', '.join([f"{k}={v}" for k, v in unifier.items()]) + '}'
                result_lines.append(f"{next_id} R[{ref1},{ref2}]{result_uni} = ({pretty_new})")
            else:
                result_lines.append(f"{next_id} R[{ref1},{ref2}] = ({pretty_new})")

            if len(new_clause) == 0:
                return result_lines

            next_id += 1
            found_new_clause = True
        else:
            break

        if not found_new_clause:
            break

    return result_lines


def format_reference(cid: int, clause: tuple, literal_index: int) -> str:
    if len(clause) == 1:
        return str(cid)
    letter = chr(ord('a') + literal_index)
    return f"{cid}{letter}"


def already_in_kb(new_clause: tuple, clauses_dict: dict) -> bool:
    new_set = set(new_clause)
    for entry in clauses_dict.values():
        if set(entry['clause']) == new_set:
            return True
    return False


def main():
    kb_line = input()
    if kb_line.startswith("kb"):
        kb_part = kb_line.split('=', 1)[1].strip()
    else:
        kb_part = kb_line
    KB = eval(kb_part)
    result = resolution_first_order(KB)
    for line in result:
        print(line)


if __name__ == "__main__":
    main()


