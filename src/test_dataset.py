from dataset import load_dataset
import matplotlib.pyplot as plt

X, y = load_dataset()

print(X.shape)
print(y.shape)

plt.subplot(1,2,1)
plt.title("RGB")
plt.imshow(X[0])

plt.subplot(1,2,2)
plt.title("Mask")
plt.imshow(y[0])

plt.show()