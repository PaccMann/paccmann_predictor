from .bimodal_mca import BimodalMCA
from .dense import Dense
from .paccmann import MCA
from .paccmann_v2 import PaccMannV2
from .knn import knn  # noqa

# More models could follow
MODEL_FACTORY = {
    'mca': MCA,
    'dense': Dense,
    'bimodal_mca': BimodalMCA,
    'paccmann_v2': PaccMannV2,
    'knn': knn
}
