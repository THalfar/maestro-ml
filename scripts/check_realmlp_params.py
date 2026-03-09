from pytabkit import RealMLP_TD_Classifier
import inspect
sig = inspect.signature(RealMLP_TD_Classifier.__init__)
for name, param in sig.parameters.items():
    if name == 'self':
        continue
    print(f'{name} = {param.default}')
