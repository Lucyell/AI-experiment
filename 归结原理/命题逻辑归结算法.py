def input_clauses():
    KB = []
    print("请输入子句集，每行一个子句，文字用逗号分隔，空行结束输入：")
    while True:
        line = input().strip()
        if not line:
            break
        clause = [lit.strip() for lit in line.split(',')]
        KB.append(clause)
    return KB


def ResolutionProp(KB):
    KB = [tuple(clause) for clause in KB]
    steps = []
    clause_entries = []
    existing_clauses = set()

    def format_clause(clause):
        return "()" if not clause else f"({','.join(clause)})"

    # 初始化处理
    for idx, clause in enumerate(KB, 1):
        clause_tuple = tuple(clause)
        clause_entries.append((idx, clause_tuple))
        existing_clauses.add(clause_tuple)
        steps.append(f"{idx}{format_clause(clause_tuple)}")

    current_max_step = len(KB)

    def is_complement(l1, l2):
        return (l1 == '~' + l2) or (l2 == '~' + l1)

    def get_letter(index):
        return chr(ord('a') + index)

    from collections import deque
    queue = deque(range(len(clause_entries)))

    while queue:
        k = queue.popleft()
        current_clause_k = clause_entries[k][1]
        current_step_k = clause_entries[k][0]

        # 修改：改为遍历所有已存在的子句
        for i in range(len(clause_entries)):
            if i == k:
                continue  # 避免同一子句内部归结

            clause_i_step, clause_i = clause_entries[i]

            # 查找互补文字对
            resolved = False
            new_elements = []  # 初始化 new_elements
            for ki, L1 in enumerate(clause_i):
                for kj, L2 in enumerate(current_clause_k):
                    if is_complement(L1, L2):
                        # 生成新子句
                        new_elements = list(clause_i[:ki]) + list(clause_i[ki + 1:])
                        new_elements += list(current_clause_k[:kj]) + list(current_clause_k[kj + 1:])
                        new_clause = tuple(sorted(set(new_elements), key=lambda x: x.strip('~')))

                        if new_clause not in existing_clauses:
                            existing_clauses.add(new_clause)
                            current_max_step += 1
                            new_step_number = current_max_step
                            clause_entries.append((new_step_number, new_clause))

                            # 生成父类标记
                            len_i = len(clause_i)
                            parent_i_letter = get_letter(ki) if len_i > 1 else ''

                            len_k = len(current_clause_k)
                            parent_k_letter = get_letter(kj) if len_k > 1 else ''

                            # 确保父类顺序稳定
                            if clause_i_step < current_step_k:
                                parents = f"{clause_i_step}{parent_i_letter},{current_step_k}{parent_k_letter}"
                            else:
                                parents = f"{current_step_k}{parent_k_letter},{clause_i_step}{parent_i_letter}"

                            step_str = f"{new_step_number} R[{parents}]={format_clause(new_clause)}"
                            steps.append(step_str)

                            # 关键修改：新子句插入队列头部优先处理
                            queue.appendleft(len(clause_entries) - 1)
                            resolved = True

                            if not new_clause:
                                return steps

                        break  # 每个子句对只处理第一组互补文字
                if resolved:
                    break

    return steps

if __name__ == "__main__":
    KB = input_clauses()
    steps = ResolutionProp(KB)
    for step in steps:
        print(step)

