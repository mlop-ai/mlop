import os
import time

import numpy as np

from .args import get_prefs, init_test, timer

TAG = "video"


@timer
def test_video(
    mlop, run, FILE_NAME=f".mlop/files/{TAG}", NUM_EPOCHS=None, ITEM_PER_EPOCH=None
):
    if NUM_EPOCHS is None or ITEM_PER_EPOCH is None:
        NUM_EPOCHS = get_prefs(TAG)["NUM_EPOCHS"]
        ITEM_PER_EPOCH = get_prefs(TAG)["ITEM_PER_EPOCH"]
    WAIT = ITEM_PER_EPOCH * 0.01
    for e in range(NUM_EPOCHS):
        epoch_time = time.time()
        examples = []
        for i in range(ITEM_PER_EPOCH):
            ndarray = mlop.Video(
                # np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8),
                np.random.randint(low=0, high=256, size=(10, 3, 112, 112), dtype=np.uint8),
                caption=f"{TAG}-{e}-{i}",
                fps=4,
            )
            examples.append(ndarray)
            run.log({f"{TAG}/data": ndarray})
        run.log({f"{TAG}/all": examples})
        print(
            f"{TAG}: Epoch {e + 1} / {NUM_EPOCHS} took {time.time() - epoch_time:.4f}s, sleeping {WAIT}s"
        )
        time.sleep(WAIT)


if __name__ == "__main__":
    mlop, run = init_test(TAG)
    test_video(mlop, run)
