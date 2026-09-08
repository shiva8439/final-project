import nibabel as nib
import matplotlib.pyplot as plt

def load_and_show(path, title="MRI"):

    img = nib.load(path)

    data = img.get_fdata()

    print("Shape :", data.shape)
    print("Data Type :", data.dtype)

    middle = data.shape[2] // 2

    plt.figure(figsize=(6,6))
    plt.imshow(data[:, :, middle], cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

    return data