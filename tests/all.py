from .args import init_test, timer
from .audio import test_audio
from .image import test_image
from .metric import test_metric

TAG = "all"


@timer
def test_all(mlop, run):
    test_metric(mlop, run)
    test_image(mlop, run)
    test_audio(mlop, run)


if __name__ == "__main__":
    mlop, run = init_test(TAG)
    test_all(mlop, run)
