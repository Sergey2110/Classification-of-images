import os
import warnings
from lib.Classification import Classification

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.simplefilter('ignore')

if __name__ == '__main__':
    classification = Classification()
    acc_loss = classification.train_model()
    classification.evaluation_model()
    classification.create_submit()
