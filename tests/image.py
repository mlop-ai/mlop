# import sys
# sys.path.append(sys.path[0] + "/..")

import os
import platform
import random
import time

import numpy as np
from PIL import Image

from .args import parse, read_sets_compat

TAG = "test-image"
ITEM_PER_EPOCH = 10
WAIT = ITEM_PER_EPOCH * 0.01
if platform.system() == "Linux":  # or platform.machine() == "x86_64"
    NUM_EPOCHS = 2  # for actions
else:
    NUM_EPOCHS = 20
TOTAL = NUM_EPOCHS * ITEM_PER_EPOCH
INIT = time.time()

FILE_NAME = ".mlop/files/image"
os.makedirs(f"{os.path.dirname(FILE_NAME)}", exist_ok=True)
IMAGE = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
IMAGE.save(f"{FILE_NAME}.png")

args = parse(TAG)
mlop, settings = read_sets_compat(args, TAG)


for i in range(TOTAL):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3))
    file = mlop.Image(image)
print(f"{TAG}: Instantiation time for {i} images: {time.time() - INIT:.4f}s")

run = mlop.init(dir=".mlop/", project=TAG, settings=settings)
# run.log({"test-image": mlop.File(".mlop/image.png")})

for e in range(NUM_EPOCHS):
    examples = []
    RUN = time.time()
    for i in range(ITEM_PER_EPOCH):
        if False:
            file = Image.new(
                "RGB",
                (100, 100),
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )
            image = f"{FILE_NAME}-{e}-{i}.png"
            file.save(image)
        else:
            image = np.random.randint(low=0, high=256, size=(100, 100, 3))

        # user shouldn't attempt to repeatedly log image with the same filename
        file = mlop.Image(image, caption=f"random-field-{e}-{i}")
        examples.append(file)
    run.log({"examples": examples})
    print(
        f"{TAG}: Epoch {e + 1} / {NUM_EPOCHS} took {time.time() - RUN:.4f}s, now waiting {WAIT}s"
    )
    time.sleep(WAIT)

print(f"{TAG}: Script time ({TOTAL}): {time.time() - INIT:.4f}s")
