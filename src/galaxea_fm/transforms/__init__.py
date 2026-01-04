class BaseActionStateTransform:
    def forward(self, batch):
        raise NotImplementedError
    
    def backward(self, batch):
        raise NotImplementedError