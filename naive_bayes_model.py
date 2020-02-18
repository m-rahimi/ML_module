x = dt[ids]; y = dtarget[ids]
x_val = dt[~ids]; y_val = dtarget[~ids]

p = x[y==1].sum(0) + 1
q = x[y==0].sum(0) + 1
r = np.log((p/p.sum())/(q/q.sum()))
b = np.log(len(q)/len(q))

pred = x_val[r.index] @ r.T + b
print('test score:', metrics.roc_auc_score(y_val, pred))
