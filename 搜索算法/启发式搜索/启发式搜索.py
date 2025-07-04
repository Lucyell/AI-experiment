import heapq
import time
import sys

es = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
cnt = 0

def mh(s):
    d = 0
    for t in range(16):
        if s[t] == es[t] or s[t] == 0:
            continue
        x = (s[t] - 1) // 4
        y = s[t] - 4 * x - 1
        dx = t // 4
        dy = t % 4
        d += abs(x - dx) + abs(y - dy)
    return d

def gc():
    mt = []
    for i in range(16):
        x, y = i % 4, i // 4
        mvs = []
        if x > 0: mvs.append(-1)
        if x < 3: mvs.append(+1)
        if y > 0: mvs.append(-4)
        if y < 3: mvs.append(+4)
        mt.append(mvs)
    def chd(s):
        iz = s.index(0)
        l = list(s)
        for m in mt[iz]:
            l[iz] = l[iz + m]
            l[iz + m] = 0
            yield (1, tuple(l))
            l[iz + m] = l[iz]
            l[iz] = 0
    return chd

class Node(object):
    def __init__(self, gn=0, hn=0, s=None, p=None):
        self.gn = gn
        self.hn = hn
        self.fn = gn + hn
        self.s = s
        self.p = p
    def __lt__(self, o):
        if self.fn == o.fn:
            return self.gn > o.gn
        return self.fn < o.fn

def a_star(s):
    global cnt
    cnt = 0
    c = set()
    o = []
    r = Node(0, mh(s), s, None)
    heapq.heappush(o, r)
    c.add(s)
    while o:
        cur = heapq.heappop(o)
        cnt += 1
        if cur.s == es:
            p = []
            while cur:
                p.append(cur.s)
                cur = cur.p
            return list(reversed(p))
        for cost, st in gc()(cur.s):
            if st in c:
                continue
            c.add(st)
            ch = Node(cur.gn + cost, mh(st), st, cur)
            heapq.heappush(o, ch)
    return None

def ida_star(s):
    global cnt
    cnt = 0
    b = mh(s)
    p = [s]
    c = set(p)
    while True:
        t = ida_s(0, b, p, c)
        if t == 0:
            return p.copy()
        if t == float('inf'):
            return None
        b = t

def ida_s(g, b, p, c):
    global cnt
    n = p[-1]
    cnt += 1
    f = g + mh(n)
    if f > b:
        return f
    if n == es:
        return 0
    m = float('inf')
    for cost, st in gc()(n):
        if st in c:
            continue
        p.append(st)
        c.add(st)
        t = ida_s(g + 1, b, p, c)
        if t == 0:
            return 0
        if t < m:
            m = t
        p.pop()
        c.remove(st)
    return m

def inp():
    while True:
        try:
            s = input("请输入初始状态(16个数字，空格分隔):\n")
            st = tuple(map(int, s.strip().split()))
            if len(st) != 16:
                print("需要输入16个数字！")
                continue
            if set(st) != set(range(16)):
                print("必须包含0-15的所有数字！")
                continue
            return st
        except:
            print("输入包含非数字字符！")

def pr_res(res, t, a):
    print(f"\n{a}结果：")
    print("总步数:", len(res) - 1)
    print("耗时: %.4f秒" % t)
    print("扩展节点数:", cnt)

def print_mat(s):
    for i in range(4):
        print(" ".join(f"{num:2d}" if num != 0 else "  " for num in s[i*4:(i+1)*4]))

if __name__ == '__main__':
    s = inp()
    print("\n请选择算法：")
    print("1. A*")
    print("2. IDA*")
    c = input()
    st = time.time()
    if c == '1':
        p = a_star(s)
        a = "A*"
    elif c == '2':
        p = ida_star(s)
        a = "IDA*"
    else:
        print("无效！")
        sys.exit()
    if p:
        pr_res(p, time.time() - st, a)
    else:
        print("无解")