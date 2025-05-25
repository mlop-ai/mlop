import math
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

    # exponential decay
    decay_rate = 0.0001
    noise_scale = 0.5

    run.alert(
        text=f"run started with {NUM_EPOCHS} epochs",
        title=TAG,
    ) if False else None

    for i in range(NUM_EPOCHS):
        base_value = math.exp(-decay_rate * i)
        run.log(
            {
                f"val/{TAG}-{j}": base_value + (random.random() - 0.5) * noise_scale
                for j in range(ITEM_PER_EPOCH)
            }
        )
        run.log({f"{TAG}-total": i + 1})
        if i % WAIT_INT == 0:
            print(f"{TAG}: Epoch {i + 1} / {NUM_EPOCHS}, sleeping {WAIT} seconds")
            time.sleep(WAIT)


if __name__ == "__main__":
    mlop, run = init_test(TAG)
    test_metric(mlop, run)
