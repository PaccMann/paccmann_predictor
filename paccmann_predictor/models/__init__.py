from .bimodal_mca import BimodalMCA
from .dense import Dense
from .paccmann import MCA
from .paccmann_v2 import PaccMannV2
from .paccmann_concentration import PaccMannConcentration
# More models could follow
MODEL_FACTORY = {
    'mca': MCA,
    'dense': Dense,
    'bimodal_mca': BimodalMCA,
    'paccmann_v2': PaccMannV2,
    'paccmann_concentration': PaccMannConcentration
}
