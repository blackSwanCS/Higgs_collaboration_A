import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def roc_curve(y_test,y_pred,weights_test_values):
    fpr_xgb,tpr_xgb,_ = roc_curve(y_true=y_test, y_score=y_pred,sample_weight=weights_test_values)
    auc_test= None
    plt.plot(fpr_xgb, tpr_xgb, color='darkgreen',lw=2, label='XGBoost (AUC  = {:.3f})'.format(auc_test))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background Efficiency')
    plt.ylabel('Signal Efficiency')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

def significance_curve(self, test_labels=None, test_weights=None):
        if test_labels is not None:
            self.__test_labels = test_labels
        if test_weights is not None:
            self.__test_weights = np.asarray(test_weights)
        if self.__status != BDT_Status.PREDICTED:
            raise ValueError(
                "Model has not been fitted or predict yet. Please call fit() and predict() before significance()."
            )
        if self.__test_labels is None:
            raise ValueError(
                "True labels for test data are not available. Please provide them when calling predict()."
            )

        def __amsasimov(s_in, b_in):
            s = np.copy(s_in)
            b = np.copy(b_in)
            s = np.where((b_in == 0), 0.0, s_in)
            b = np.where((b_in == 0), 1.0, b)
            ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
            ams = np.where((s < 0) | (b < 0), np.nan, ams)
            if np.isscalar(s_in):
                return float(ams)
            else:
                return ams

        def __significance_vscore(y_true, y_score, sample_weight=None):
            if sample_weight is None:
                sample_weight = np.full(len(y_true), 1.0)
            else:
                sample_weight = np.asarray(sample_weight)
            bins = np.linspace(0, 1.0, 101)
            s_hist, bin_edges = np.histogram(
                y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
            )
            b_hist, bin_edges = np.histogram(
                y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
            )
            print(s_hist, b_hist)
            s_cumul = np.cumsum(s_hist[::-1])[::-1]
            b_cumul = np.cumsum(b_hist[::-1])[::-1]
            significance = __amsasimov(s_cumul, b_cumul)
            return significance

        vamsasimov = __significance_vscore(
            y_true=self.__test_labels,
            y_score=self.__predicted_data,
            sample_weight=self.__test_weights,
        )
        x = np.linspace(0, 1, num=len(vamsasimov))
        significance=np.max(vamsasimov)
        

        plt.plot(x, vamsasimov,label=f'{self.name} '+'(Z = {:.2f})'.format(significance))


plt.title("BDT Significance")
plt.xlabel("Threshold")
plt.ylabel("Significance")
plt.legend()
plt.savefig("Significance_comparing.pdf")
plt.show()