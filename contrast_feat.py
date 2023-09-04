import numpy as np

from scipy.stats import entropy


def JS_divergence(pVect1,pVect2):
    logQvect=np.log2((pVect1+pVect2)/2)
    KL = 0.5 * (np.sum(np.dot(pVect1 , (np.log2(pVect1) - logQvect  ) ) ) +
                np.sum(np.dot(pVect2 , (np.log2(pVect2) - logQvect ) ) ) )
    return KL

def contrast_feat(im):
    qq = np.arange(0, 258, 2)

    rr, _ = np.histogram(im.flatten(), bins=qq,range=(im.max(),im.min()))

    rr = (1 + 2 * rr) / np.sum(1 + 2 * rr)
    uu = np.ones_like(qq[:128]) / np.sum(np.ones_like(qq[:128]))


    gg = JS_divergence(rr, uu)

    return gg

