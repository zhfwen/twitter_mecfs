import shap
import pickle

def get_shap_scores(args):
    fpath, label, dataset, id_min, id_max = args
    with open(fpath, "rb") as f:
        sv = pickle.load(f)
        
    features = []
    values = []
    base_values = []
    ids = []
    
    sv_class = sv[:, :, label]
    
    for i in range(len(sv_class)):
        id = id_min + i
        if id > id_max:
            raise Exception ("wrong tweet id")
        if len(sv_class.data[i]) != len(sv_class.values[i]):
            raise Exception("length does not match")
        
        features.extend(list(sv_class.data[i]))
        values.extend(list(sv_class.values[i]))
        base_values.extend([sv_class.base_values[i]] * len(list(sv_class.data[i])))
        ids.extend([id] * len(list(sv_class.data[i])))
        
    return features, values, base_values, ids, label, dataset
