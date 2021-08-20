from sklearn.metrics import mean_absolute_error, mean_squared_error

def metrics(y, ypred, y_preprocessor=None):
    """Evaluates metrics.
    
    Args:
        y:
        ypred:
        y_scaler:
        
    Returns:
        mae: mean absolute error, float scaler.
        std_err: standard error, float scaler.
    """
    if y_preprocessor is not None:
        y = y_preprocessor.inverse_transform(y)
        ypred = y_preprocessor.inverse_transform(ypred)
    
    mse = mean_squared_error(y, ypred)
    rmse = mean_squared_error(y, ypred, squared=False)
    mae = mean_absolute_error(y, ypred)
    std_err = (mean_squared_error(y, ypred)**0.5)/(y.shape[0]**0.5)
    return mse, rmse, mae, std_err
    