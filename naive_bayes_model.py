#https://www.aclweb.org/anthology/P12-2018.pdf

x = dt[ids]; y = dtarget[ids]
x_val = dt[~ids]; y_val = dtarget[~ids]

p = x[y==0].sum(0) + 1
q = x[y==1].sum(0) + 1
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log((y==1).mean() / (y==0).mean())

pred = x_val[r.index] @ r.T + b
print('test score:', metrics.roc_auc_score(y_val, pred))
