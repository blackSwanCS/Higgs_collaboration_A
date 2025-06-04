import xgb_boosted_decision_tree
import constants

def get_best_model():
    """
        Returns the best pre-trained Boosted Decision Tree model we found so far
    """
    model = xgb_boosted_decision_tree.XGBBoostedDecisionTree()
    model.load_model(constants.BEST_BDT_MODEL_PATH)
    return model
    #params = {'n_estimators': np.int64(236), 'max_depth': np.int64(9), 'max_leaves': np.int64(0), 'objective': 'binary:logistic', 'use_label_encoder': False, 'eval_metric': 'logloss'}
    #return XGBBoostedDecisionTree(params)
