from .bimodal_mca import BimodalMCA
from .dense import Dense
from .paccmann import MCA

# More models could follow
MODEL_FACTORY = {'mca': MCA, 'dense': Dense, 'bimodal_mca': BimodalMCA}
