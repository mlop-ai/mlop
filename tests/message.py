import sys
import time

from .args import parse, read_sets_compat

TAG = "test-message"
TOTAL = 100
INIT = time.time()

args = parse(TAG)
mlop, settings = read_sets_compat(args, TAG)



run = mlop.init(dir=".mlop/", project=TAG, settings=settings)
print(f"{TAG}: Init time: {time.time() - INIT:.4f}s")

for i in range(TOTAL):
    print(f"{TAG}: Epoch {i + 1} / {TOTAL})")
    sys.stderr.write(f"{TAG}: Err {i + 1} / {TOTAL}\n")

print(f"{TAG}: Script time ({TOTAL}): {time.time() - INIT:.4f}s")
