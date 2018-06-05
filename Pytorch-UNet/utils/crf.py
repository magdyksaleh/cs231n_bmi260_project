import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral


def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    print(U.shape)
    U = U.reshape((2, -1))
    print(U.shape)
    U = np.ascontiguousarray(U)
    # img = np.ascontiguousarray(img)


    print("img shape: ", img.shape)

    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=20, compat=3)
    energy = create_pairwise_bilateral(sdims=(10,10), schan=0.01, img=img)
    d.addPairwiseEnergy(energy, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
