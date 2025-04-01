import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .args import get_prefs, init_test, timer

TAG = "image"


def gen_mpl_image(size: tuple = (100, 100)):
    noise = np.random.rand(*size)
    x = np.linspace(0, 1, size[1])
    y = np.linspace(0, 1, size[0])
    X, Y = np.meshgrid(x, y)
    gradient = X + Y
    image = (noise + gradient) / 2
    fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
    ax.imshow(image, cmap="viridis")
    ax.axis("off")
    return fig, ax


def gen_pil_image(size: tuple = (100, 100)):
    return Image.new(
        "RGB",
        size,
        (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ),
    )


@timer
def test_image(
    mlop, run, FILE_NAME=f".mlop/files/{TAG}", NUM_EPOCHS=None, ITEM_PER_EPOCH=None
):
    os.makedirs(f"{os.path.dirname(FILE_NAME)}", exist_ok=True)

    if NUM_EPOCHS is None or ITEM_PER_EPOCH is None:
        NUM_EPOCHS = get_prefs(TAG)["NUM_EPOCHS"]
        ITEM_PER_EPOCH = get_prefs(TAG)["ITEM_PER_EPOCH"]
    WAIT = ITEM_PER_EPOCH * 0.01

    inst_time = time.time()
    for i in range(NUM_EPOCHS * ITEM_PER_EPOCH):
        image = np.random.randint(low=0, high=256, size=(100, 100, 3))
        _ = mlop.Image(image)
    print(f"{TAG}: Instantiation time for {i} images: {time.time() - inst_time:.4f}s")

    for e in range(NUM_EPOCHS):
        epoch_time = time.time()
        for i in range(ITEM_PER_EPOCH):
            fp = f"{FILE_NAME}-{e}-{i}.png"
            pil_img = gen_pil_image()
            pil_img.save(fp)
            mpl_img, _ = gen_mpl_image()
            np_img = np.random.randint(low=0, high=256, size=(100, 100, 3))

            images = [
                ("file", mlop.Image(fp, caption=f"file-{e}-{i}")),
                ("pil", mlop.Image(pil_img, caption=f"pil-{e}-{i}")),
                ("np", mlop.Image(np_img, caption=f"np-{e}-{i}")),
                # TODO: make mpl more efficient
                ("mpl", mlop.Image(mpl_img, caption=f"mpl-{e}-{i}")),
            ]

            examples = [img for _, img in images]
            for img_type, img in images:
                run.log({f"{TAG}/{img_type}/{e}-{i}": img})
            run.log({f"{TAG}/all": examples})

        print(
            f"{TAG}: Epoch {e + 1} / {NUM_EPOCHS} took {time.time() - epoch_time:.4f}s, sleeping {WAIT}s"
        )
        time.sleep(WAIT)


if __name__ == "__main__":
    mlop, run = init_test(TAG)
    test_image(mlop, run)
