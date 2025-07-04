import numpy as np
import matplotlib.pyplot as plt
import random
import time
from itertools import combinations
from multiprocessing import Pool

class TSPSolver:
    def __init__(self, f):
        self.ct = self.rd(f)
        self.dm = self.cm()
        self.n = len(self.ct)
        self.br = None
        self.bd = float('inf')
        self.hist = []
        
    def rd(self, fp):
        ct = []
        with open(fp, 'r') as f:
            cs = False
            for l in f:
                if l.startswith("NODE_COORD_SECTION"):
                    cs = True
                    continue
                if l.startswith("EOF"):
                    break
                if cs:
                    p = l.strip().split()
                    try:
                        ct.append((float(p[1]), float(p[2])))
                    except:
                        continue
        return np.array(ct)
    
    def cm(self):
        if hasattr(self, 'sd'):
            return self.sd()
        
        d = self.ct[:, np.newaxis] - self.ct[np.newaxis, :]
        return np.sqrt(np.sum(d**2, axis=2))
    
    def rdist(self, r):
        return np.sum(self.dm[r[:-1], r[1:]]) + self.dm[r[-1], r[0]]
    
    def ip(self, sz):
        p = []
        p += [np.random.permutation(self.n).tolist() for _ in range(sz//2)]
        for _ in range(sz//2):
            s = random.randint(0, self.n-1)
            r = [s]
            uv = set(range(self.n)) - {s}
            while uv:
                nc = min(uv, key=lambda x: self.dm[r[-1], x])
                r.append(nc)
                uv.remove(nc)
            p.append(r)
        return p
    
    def erx(self, p1, p2):
        et = {i: set() for i in range(self.n)}
        for p in [p1, p2]:
            for i in range(self.n):
                l = p[(i-1)%self.n]
                r = p[(i+1)%self.n]
                et[p[i]].update({l, r})
        
        cur = random.choice([p1[0], p2[0]])
        r = [cur]
        for _ in range(self.n-1):
            nb = et[cur] - set(r)
            if not nb:
                ch = list(set(range(self.n)) - set(r))
            else:
                ch = list(nb)
            cur = min(ch, key=lambda x: len(et[x] - set(r)))
            r.append(cur)
        return r
    
    def mut(self, r, mr):
        if random.random() < mr:
            i, j = sorted(random.sample(range(self.n), 2))
            r[i:j+1] = r[i:j+1][::-1]
        if random.random() < mr:
            s, d = random.sample(range(self.n), 2)
            c = r.pop(s)
            r.insert(d, c)
        return r
    
    def to(self, r):
        imp = True
        while imp:
            imp = False
            for i, j in combinations(range(1, len(r)-1), 2):
                if j <= i: continue
                od = (self.dm[r[i-1], r[i]] + self.dm[r[j], r[j+1]])
                nd = (self.dm[r[i-1], r[j]] + self.dm[r[i], r[j+1]])
                if nd < od:
                    r[i:j+1] = r[i:j+1][::-1]
                    imp = True
                    break
        return r
    
    def ap(self, g, mg):
        mr = 0.2 * (1 - g/mg)**2 + 0.05
        cr = 0.9 * (0.5 + 0.5*g/mg)
        return mr, cr
    
    def run(self, ps=500, mg=2000, wk=4):
        pl = Pool(wk)
        try:
            pop = self.ip(ps)
        
            for g in range(mg):
                ts = [(r,) for r in pop]
                ds = pl.starmap(self.rdist, ts)
                
                mi = np.argmin(ds)
                cb = ds[mi]
                if cb < self.bd:
                    self.bd = cb
                    self.br = pop[mi].copy()
                self.hist.append(self.bd)
                
                mr, cr = self.ap(g, mg)
                
                sl = []
                ts = 5
                for _ in range(ps):
                    cd = random.sample(list(zip(pop, ds)), ts)
                    wn = min(cd, key=lambda x: x[1])[0]
                    sl.append(wn.copy())
                
                npop = []
                for i in range(0, ps, 2):
                    if random.random() < cr:
                        c1 = self.erx(sl[i], sl[i+1])
                        c2 = self.erx(sl[i+1], sl[i])
                    else:
                        c1, c2 = sl[i].copy(), sl[i+1].copy()
                    npop.extend([c1, c2])
                
                pop = [self.mut(ind, mr) for ind in npop[:ps]]
                
                es = ps//10
                el = sorted(zip(pop, ds), key=lambda x: x[1])[:es]
                for i in range(es):
                    imp = self.to(el[i][0].copy())
                    if self.rdist(imp) < el[i][1]:
                        pop.append(imp)
                
                pop[-es:] = [self.br.copy() for _ in range(es)]
                
                if len(set(tuple(r) for r in pop)) < ps//2:
                    pop = pop[:ps//2] + self.ip(ps//2)
                
                if g % 100 == 0:
                    print(f"Gen {g}: Best {self.bd:.2f} | Diversity {len(set(map(tuple, pop)))}")
        
        finally:
            pl.close()
    
    def plot(self):
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        od = self.ct[self.br + [self.br[0]]]
        plt.plot(od[:,0], od[:,1], 'b-')
        plt.scatter(self.ct[:,0], self.ct[:,1], c='r')
        plt.title(f"Optimal Route: {self.bd:.2f}")
        
        plt.subplot(122)
        plt.plot(self.hist)
        plt.title("Convergence History")
        plt.xlabel("Generation")
        plt.ylabel("Distance")
        plt.show()

if __name__ == "__main__":
    solver = TSPSolver("wi29.tsp")
    st = time.time()
    solver.run(ps=200, mg=2000)
    print(f"Total time: {time.time()-st:.2f}s")
    solver.plot()