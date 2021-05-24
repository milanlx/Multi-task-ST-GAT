import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date


# -------------------------- save and load -------------------------- #
def saveAsPickle(obj, pickle_file):
    with open(pickle_file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()


def loadFromPickle(pickle_file):
    with open(pickle_file, 'rb') as handle:
        unserialized_obj = pickle.load(handle)
    handle.close()
    return unserialized_obj