"""Copy into a new hypothesis directory and implement a measurable score."""


class ResearchDetector:
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, device

    def score(self, images):
        raise NotImplementedError("先明确待验证的机制，再实现每个样本的异常分数")
