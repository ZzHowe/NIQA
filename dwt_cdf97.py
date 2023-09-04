import numpy as np
from scipy.signal import lfilter


def dwt_cdf97(X, Level):
    if Level < 0:
        raise ValueError("Invalid transform level.")
    N1, N2 = X.shape[0], X.shape[1]

    # Lifting scheme filter coefficients for CDF 9/7
    LiftFilter = np.array([-1.5861343420693648, -0.0529801185718856, 0.8829110755411875, 0.4435068520511142])
    ScaleFactor = 1.1496043988602418
    S1, S2, S3 = LiftFilter[0], LiftFilter[1], LiftFilter[2]
    ExtrapolateOdd = -2 * np.array([S1 * S2 * S3, S2 * S3, S1 + S3 + 3 * S1 * S2 * S3]) / (1 + 2 * S2 * S3)
    LiftFilter = LiftFilter[np.newaxis, :]
    LiftFilter = np.repeat(LiftFilter, 2, axis=0)

    if Level >= 0:
        for k in range(Level):
            M1 = int(np.ceil(N1 / 2))
            M2 = int(np.ceil(N2 / 2))
            print(M1)
            if N1 > 1:
                RightShift = np.arange(2, M1 + 1)
                RightShift = np.append(RightShift, M1)
                X0 = np.transpose(X[range(0, N1, 2), 0:N2], (0, 1))
                print("X0.shape: ", X0.shape)
                # Apply lifting stages
                if N1 % 2:
                    part1 = np.hstack((X[range(1, N1, 2), 0:N2], np.array([X0[M1 - 2, :] * ExtrapolateOdd[0] + X[N1 - 2,
                                                                                                               0:N2] *
                                                                           ExtrapolateOdd[1] + X0[M1 - 1, :] *
                                                                           ExtrapolateOdd[2]])))
                    part2 = (lfilter(LiftFilter[:, 0], np.array([1]), (X0[RightShift - 1, :]).ravel(),
                                     X0[0, :] * LiftFilter[0, 0],0,))[0]
                    X1 = part1 + part2.reshape(part2.shape[0], 1)
                else:
                    part1 = X[range(1, N1, 2), 0:N2]
                    part2 = (lfilter(LiftFilter[:, 0], np.array([1]), (X0[RightShift - 1, :]).ravel(),
                                     X0[0, :] * LiftFilter[0, 0],0))[0]
                    X1 = part1 + part2.reshape(part2.shape[0], 1)

                part3 = (lfilter(LiftFilter[:, 1], np.array([1]), X1.ravel(), 0, X1[0, :] * LiftFilter[0, 1]))[0]
                X0 = X0 + part3.reshape(part3.shape[0], 1)
                part4 = (lfilter(LiftFilter[:, 2], np.array([1]), (X0[RightShift - 1, :]).ravel(),
                                 X0[0, :] * LiftFilter[0, 2],0))[0]
                X1 = X1 + part4.reshape(part4.shape[0], 1)
                part5 = (lfilter(LiftFilter[:, 3], np.array([1]), X1.ravel(), X1[0, :] * LiftFilter[0, 3],0))[0]
                X0 = X0 + part5.reshape(part5.shape[0], 1)
                if N1 % 2 != 0:
                    X1 = np.delete(X1, M1 - 1, axis=0)

                X[0:N1, 0:X0.shape[0]] = X0 * ScaleFactor
                X[0:N1, X0.shape[0]:N2] = X1 / ScaleFactor

            # Transform along rows
            if N2 > 1:
                RightShift = np.arange(2, M2 + 1)
                RightShift = np.append(RightShift, M2)
                X0 = np.transpose(X[0:N1, range(0, N2, 2)], (1, 0))
                print("X0.shape: ", X0.shape)

                # Apply lifting stages
                if N2 % 2:
                    part1 = np.transpose(np.hstack((X[0:N1, range(1, N2, 2)], np.array([X[0:N1, N2 - 3] *
                                                                                        ExtrapolateOdd[0] + X[0:N1,
                                                                                                            N2 - 2] *
                                                                                        ExtrapolateOdd[1] + X[0:N1,
                                                                                                            N2 - 1] *
                                                                                        ExtrapolateOdd[2]]))), (1, 0))
                    part2 = (lfilter(LiftFilter[:, 0], np.array([1]), (X0[RightShift - 1, :]).ravel(),
                                     X0[0, :] * LiftFilter[0, 0],0))[0]
                    X1 = part1 + part2.reshape(part2.shape[0], 1)
                else:
                    part1 = np.transpose(X[0:N1, range(1, N2, 2)], (1, 0))
                    part2 = (lfilter(LiftFilter[:, 0], np.array([1]), (X0[RightShift - 1, :]).ravel(),
                                     X0[0, :] * LiftFilter[0, 0],0))[0]
                    X1 = part1 + part2.reshape(part2.shape[0], 1)

                part3 = (lfilter(LiftFilter[:, 1], np.array([1]), X1.ravel(),X1[0, :] * LiftFilter[0, 1],0))[0]
                X0 = X0 + part3.reshape(part3.shape[0], 1)
                part4 = (lfilter(LiftFilter[:, 2], np.array([1]), (X0[RightShift - 1, :]).ravel(), 0,
                                 X0[0, :] * LiftFilter[0, 2]))[0]
                X1 = X1 + part4.reshape(part4.shape[0], 1)
                part5 = (lfilter(LiftFilter[:, 3], np.array([1]), X1.ravel(), X1[0, :] * LiftFilter[0, 3],0))[0]
                X0 = X0 + part5.reshape(part5.shape[0], 1)
                if N2 % 2 != 0:
                    X1 = np.delete(X1, M2 - 1, axis=0)

                X[0:N1, 0:X0.shape[0]] = np.transpose(X0 * ScaleFactor, (1, 0))
                X[0:N1, X0.shape[0]:N2] = np.transpose(X1 / ScaleFactor, (1, 0))

            N1 = np.copy(M1)
            N2 = np.copy(M2)

        N1, N2 = X.shape[0], X.shape[1]
        bands = []
        for lev in range(Level, 0, -1):
            k = 2 ** lev
            if lev == Level:
                bands_lev = [None] * 4
                bands_lev[3] = X[0:N1 // k, 0:N2 // k]  # LL
            else:
                bands_lev = [None] * 3
            bands_lev[0] = X[0:N1 // k, N2 // k:2 * N2 // k]  # LH
            bands_lev[1] = X[N1 // k:2 * N1 // k, 0:N2 // k]  # HL
            bands_lev[2] = X[N1 // k:2 * N1 // k, N2 // k:2 * N2 // k]  # HH

            bands.append(bands_lev)
    return bands
