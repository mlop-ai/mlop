import platform
import random
import time

from .args import parse, read_sets_compat

TAG = "test-metric"
if platform.system() == "Linux":
    NUM_EPOCHS = 10  # for actions
    ITEM_PER_EPOCH = 10
else:
    # TODO: for server to catch up and sync up live, may potentially tell server to only respond a status code when the entire batch has been processed in the backend db
    NUM_EPOCHS = 50_000
    ITEM_PER_EPOCH = 20
WAIT_INT = 1_000
WAIT = ITEM_PER_EPOCH * 0.0005
TOTAL = NUM_EPOCHS * ITEM_PER_EPOCH
INIT = time.time()

args = parse(TAG)
mlop, settings = read_sets_compat(args, TAG)



run = mlop.init(dir=".mlop/", project=TAG, settings=settings)
print(f"{TAG}: Init time: {time.time() - INIT:.4f}s")

BLOCK = time.time()
run.log({"train-loss": random.random()})
print(f"{TAG}: Blocking time: {time.time() - BLOCK:.4f}s")

for i in range(NUM_EPOCHS):
    dummy_data = {f"val/metric-{i}": random.random() for i in range(ITEM_PER_EPOCH)}
    run.log(dummy_data)

    if i % WAIT_INT == 0:
        print(f"{TAG}: Epoch {i + 1} / {NUM_EPOCHS}, sleeping {WAIT}s")
        time.sleep(WAIT)

print(f"{TAG}: Script time ({TOTAL}): {time.time() - INIT:.4f}s")
