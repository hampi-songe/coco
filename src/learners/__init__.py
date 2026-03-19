from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .coco_learner import COCOLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coco_learner"] = COCOLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
