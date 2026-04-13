
x = [1, 2, 3]
y = [4, 7, 10]

w = 1
b = 0
lr = 0.01

for i in range(1000):
    totalloss = 0
    dw_sum = 0
    db_sum = 0
    for xi, yi in zip(x, y):
        y_pred = w * xi + b
        loss = (y_pred - yi) ** 2
        dw = 2 * xi * (y_pred - yi)
        db = 2 * (y_pred - yi)

        totalloss += loss
        dw_sum += dw
        db_sum += db
    w = w - lr * (dw_sum / 3)
    b = b - lr * (db_sum / 3)

print(f"最终w:{w:6f}   最终b:{b:6f}")
