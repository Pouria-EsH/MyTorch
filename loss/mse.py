from mytorch import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    se = (preds - actual)**2
    return se * (1/ se.data.size)
