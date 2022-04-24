import math


class choose_granularity_alpha:
    def __init__(self):
        pass

    def get_m_1(self, ep=None, n1=None, alpha_1=None):
        e_ep = math.exp(ep)
        tmp = 2 * alpha_1 * (e_ep - 1) * math.sqrt(n1 / e_ep)
        m1 = math.sqrt(tmp)
        if m1 == 0:
            m1 = 1
        return math.ceil(m1)

    def get_m_2(self, ep=None, n2=None, f_k=None, alpha_2=None):
        e_ep = math.exp(ep)
        tmp = 2 * alpha_2 * f_k * (e_ep - 1) * math.sqrt(n2 / e_ep)
        m2 = math.sqrt(tmp)
        if m2 == 0:
            m2 = 1
        return math.ceil(m2)
