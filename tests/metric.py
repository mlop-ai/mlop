import random
import time

from .args import get_prefs, init_test, timer

TAG = "metric"


@timer
def test_metric(mlop, run, NUM_EPOCHS=None, ITEM_PER_EPOCH=None):
    if NUM_EPOCHS is None or ITEM_PER_EPOCH is None:
        NUM_EPOCHS = get_prefs(TAG)["NUM_EPOCHS"]
        ITEM_PER_EPOCH = get_prefs(TAG)["ITEM_PER_EPOCH"]

    WAIT_INT = 1_000
    WAIT = ITEM_PER_EPOCH * 0.0005

    for i in range(NUM_EPOCHS):
        run.log({f"val/{TAG}-{i}": random.random() for i in range(ITEM_PER_EPOCH)})
        run.log({f"val/{TAG}-total": i + 1})
        if i % WAIT_INT == 0:
            print(f"{TAG}: Epoch {i + 1} / {NUM_EPOCHS}, sleeping {WAIT}s")
            time.sleep(WAIT)


if __name__ == "__main__":
    mlop, run = init_test(TAG)
    test_metric(mlop, run)
