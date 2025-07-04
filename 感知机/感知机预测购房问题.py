import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

data = pd.read_csv('MLP_data.csv')
x = data[['housing_age', 'homeowner_income']].values
y = data['house_price'].values.reshape(-1, 1)
seed = int(time.time()) % 10000

def split(x, y, tr=0.7, v=0.15):
    n = x.shape[0]
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    x, y = x[idx], y[idx]
    te = int(n * tr)
    ve = te + int(n * v)

    return x[:te], y[:te], x[te:ve], y[te:ve], x[ve:], y[ve:]

x_tr, y_tr, x_val, y_val, x_te, y_te = split(x, y)

mean = x_tr.mean(axis=0)
std = x_tr.std(axis=0)

def norm(x):
    return (x - mean) / std

x_tr = norm(x_tr)
x_val = norm(x_val)
x_te = norm(x_te)

lr = 0.005
epochs = 500
l2 = 0.05
mom = 0.9
decay = 0.992

np.random.seed(0)
w = np.random.randn(2, 1)
b = np.random.randn(1)
vw = np.zeros_like(w)
vb = np.zeros_like(b)

def lr_sched(lr, epoch):
    return lr * (decay ** epoch)

tr_loss = []
val_loss = []
cur_lr = lr

for e in range(epochs):
    yp = np.dot(x_tr, w) + b
    
    mse = np.mean((yp - y_tr) ** 2)
    reg = (l2 / (2 * x_tr.shape[0])) * np.sum(w ** 2)
    loss = mse + reg
    tr_loss.append(mse)
    
    yv = np.dot(x_val, w) + b
    vl = np.mean((yv - y_val) ** 2)
    val_loss.append(vl)
    
    dw = (2/x_tr.shape[0]) * np.dot(x_tr.T, (yp - y_tr)) + (l2/x_tr.shape[0])*w
    db = (2/x_tr.shape[0]) * np.sum(yp - y_tr)
    
    vw = mom * vw + dw
    vb = mom * vb + db
    
    cur_lr = lr_sched(lr, e)
    w -= cur_lr * vw
    b -= cur_lr * vb

plt.plot(tr_loss, label="train")
plt.plot(val_loss, label="val")
plt.xlabel("epoch")
plt.ylabel("mse")
plt.title("loss curve")
plt.legend()
plt.grid(True)
plt.show()

def mse(yt, yp):
    return np.mean((yt - yp) ** 2)

def rmse(yt, yp):
    return np.sqrt(mse(yt, yp))

def mae(yt, yp):
    return np.mean(np.abs(yt - yp))

y_te_pred = np.dot(x_te, w) + b
r = rmse(y_te, y_te_pred)
m = mae(y_te, y_te_pred)

print(f"test rmse: {r:.4f}")
print(f"test mae : {m:.4f}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ha = x_te[:, 0] * std[0] + mean[0]
hi = x_te[:, 1] * std[1] + mean[1]
tp = y_te.ravel()

ax.scatter(ha, hi, tp, color='blue', label='true')

gx, gy = np.meshgrid(
    np.linspace(ha.min(), ha.max(), 10),
    np.linspace(hi.min(), hi.max(), 10)
)
gi = np.stack([
    (gx - mean[0]) / std[0],
    (gy - mean[1]) / std[1]
], axis=-1).reshape(-1, 2)
gz = np.dot(gi, w).reshape(gx.shape) + b

ax.plot_surface(gx, gy, gz, color='green', alpha=0.5)
ax.set_xlabel('housing_age')
ax.set_ylabel('homeowner_income')
ax.set_zlabel('house_price')
plt.title('prediction surface')
plt.legend()
plt.show()