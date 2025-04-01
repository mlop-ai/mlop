import os
import time

import httpx

from .args import get_prefs, init_test, timer

TAG = "audio"


def gen_audio(FILE_NAME=f".mlop/files/{TAG}"):
    os.makedirs(f"{os.path.dirname(FILE_NAME)}", exist_ok=True)
    r = httpx.get(
        "https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg"
    )
    with open(f"{FILE_NAME}.ogg", "wb") as f:
        f.write(r.content)


@timer
def test_audio(
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
            file = mlop.Audio(f"{FILE_NAME}.ogg", caption=f"{TAG}-{e}-{i}")
            examples.append(file)
            run.log({f"{TAG}/file": file})
        run.log({f"{TAG}/all": examples})
        print(
            f"{TAG}: Epoch {e + 1} / {NUM_EPOCHS} took {time.time() - epoch_time:.4f}s, sleeping {WAIT}s"
        )
        time.sleep(WAIT)


if __name__ == "__main__":
    mlop, run = init_test(TAG)
    test_audio(mlop, run)
