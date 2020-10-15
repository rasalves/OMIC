import numpy as np

class OMIC:
    def __init__(self, R,Xs,Ys,Lambdas, epochs=100,thresh=10^-5):
        self.R = R
        self.Xs = Xs
        self.Ys = Ys
        self.Lambdas = Lambdas
        self.epochs = epochs
        self.thresh = thresh
        self.m = R.shape[0]
        self.n = R.shape[1]

	######################################
	# Useful methods to create auxiliary matrices
	######################################
    @staticmethod
    def complement(m):
        X = np.array(np.repeat(0.0, m * (m - 1))).reshape(m, m - 1)
        for i in range((m - 2)):
            v = X[:, i]
            v[0:i] = 0
            v[i] = 1
            v[(i + 1):m] = (-1 / (m - i - 1))
            X[:, i] = v / np.linalg.norm(v)
        v = X[:, m - 2]
        v[m - 2] = 1
        v[m - 1] = -1
        X[:, m - 2] = v / np.linalg.norm(v)
        return X

    @staticmethod
    def complementVector(v):
        x = v.copy()
        m = v.shape[0]
        comp = OMIC.complement(np.sum(x != 0))
        Xcomp = np.array(np.repeat(0.0, m * (comp.shape[1]))).reshape(m, comp.shape[1])
        Xcomp[x != 0, :] = comp
        return Xcomp

    @staticmethod
    def complementBias(v):
        mx = int(np.max(v))
        cls = np.repeat(0.0, mx + 1)
        for i in range(1, (mx + 1)):
            cls[i] = np.sum(v == i)
        m = len(v)
        X = np.array(np.repeat(0.0, m * (mx - 1))).reshape(m, mx - 1)
        for i in range(1, (mx)):
            cv = np.concatenate((np.repeat(0.0, np.sum(cls[range(0, i)])), np.repeat(1 / cls[i], cls[i])), axis=None)
            cv = np.concatenate((cv, np.repeat((-1 / (np.sum(cls[(i + 1):(mx + 1)]))), np.sum(cls[(i + 1):(mx + 1)]))),
                                axis=None)
            X[:, i - 1] = np.transpose((cv / np.linalg.norm(cv))[np.newaxis])
        return X

    @staticmethod
    def getBomicSI(m):
        X1 = (np.repeat(1 / np.sqrt(m), m)).reshape(m, 1)
        X2 = OMIC.complementVector(np.repeat(1 / np.sqrt(m), m))
        return [X1, X2]
	######################################
	# Useful methods to generate synthetic matrices
	######################################
    def genSubMatrix(m, n, q):
        return np.matrix(
            np.random.choice(np.concatenate((np.repeat(True, (m * n) - q), np.repeat(False, q)), axis=0), n * m,
                             False)).reshape(m, n)

    @staticmethod
    def genMatrix(m, n, gamma, alpha, Pomega):
        S1_1 = np.repeat(1 / np.sqrt(m), m)
        S1_2 = np.repeat(1 / np.sqrt(n), n)
        S2_1 = (np.arange(m) + 1) - np.mean((np.arange(m) + 1))
        S2_1 = (S2_1) / np.linalg.norm(S2_1)
        S2_2 = (np.arange(n) + 1) - np.mean((np.arange(n) + 1))
        S2_2 = (S2_2) / np.linalg.norm(S2_2)
        S3_1 = np.concatenate((np.repeat(-1, m / 2), np.repeat(1, m / 2)), axis=0)
        S3_1 = (S3_1) / np.linalg.norm(S3_1)
        S3_2 = S3_1
        S1_1 = S1_1[np.newaxis]
        S1_2 = S1_2[np.newaxis]
        S2_1 = S2_1[np.newaxis]
        S2_2 = S2_2[np.newaxis]
        S3_1 = S3_1[np.newaxis]
        S3_2 = S3_2[np.newaxis]
        XFull = ((n * (1 - alpha)) * ((0.5 * (S2_1.transpose() @ S1_2)) + (0.5 * (S1_1.transpose() @ S2_2)))) + (
                (n * alpha) * (S3_1.transpose() @ S3_2))
        X = XFull.copy()
        q1 = (1 / (gamma + 1)) * gamma / 2
        q2 = (1 / (gamma + 1)) / 2
        (X[0:(m // 2), 0:(n // 2)])[OMIC.genSubMatrix(m // 2, n // 2, q1 * (m * n * Pomega))] = np.nan
        (X[0:(m // 2), (n // 2):n])[OMIC.genSubMatrix(m // 2, n // 2, q1 * (m * n * Pomega))] = np.nan
        (X[(m // 2):m, 0:(n // 2)])[OMIC.genSubMatrix(m // 2, n // 2, q2 * (m * n * Pomega))] = np.nan
        (X[(m // 2):m, (n // 2):n])[OMIC.genSubMatrix(m // 2, n // 2, q2 * (m * n * Pomega))] = np.nan
        return [X,XFull]

	######################################
	# Auxiliaries and main OMIC algorithm
	######################################
    @staticmethod
    def normConv(R, NR):
        return (((np.linalg.norm(NR - R, 'fro')) ** 2) / ((np.linalg.norm(R, 'fro')) ** 2))

    @staticmethod
    def svdThreshold(X, lam):
        u, d, v = np.linalg.svd(X)
        d = (d[d - lam > 0]) - lam
        if d.shape[0] == 0:
            S = X.copy()
            S[:, :] = 0
            return S
        else:
            return u[:, range(d.shape[0])] @ np.diag(d) @ v[range(d.shape[0]), :]

    def run(self):
        Z = self.R.copy()
        Z[np.isnan(R)] = 0.0
        for i in range(self.epochs):
            Ms = []
            for k in range(len(self.Xs)):
                for l in range(len(self.Ys)):
                    Ms.append(OMIC.svdThreshold(np.transpose(self.Xs[k]) @ Z @ self.Ys[l], Lambdas[k, l]))
                    if k == 0 and l == 0:
                        NZ = self.Xs[k] @ Ms[len(Ms) - 1] @ np.transpose(self.Ys[l])
                    else:
                        NZ = NZ + self.Xs[k] @ Ms[len(Ms) - 1] @ np.transpose(self.Ys[l])
            if OMIC.normConv(Z, NZ) < self.thresh:
                self.iterations = i
                self.Ms = Ms
                Z = NZ
                Z[~np.isnan(R)] = R[~np.isnan(R)]
                self.Z = Z
                self.converged = True
                return True
            Z = NZ
            Z[~np.isnan(R)] = R[~np.isnan(R)]
        self.iterations = i
        self.Ms = Ms
        self.Z = Z
        self.converged = True
        return False
	
	######################################
	# Get Methods
	######################################
    def getNumberOfIterations(self):
        return self.iterations

    def getMs(self):
        return self.Ms

    def isConverged(self):
        return self.converged

    def getRecoveredMatrix(self):
        return self.Z

######################################
# Toy example of BOMIC
######################################
if __name__ == "__main__":
    m=100
    n=100

    [R, R_FULL] = OMIC.genMatrix(m,n,1,0.7,0.5)
    Lambdas=np.array([0,0.5,0.5,1]).reshape(2,2)

    Xs = OMIC.getBomicSI(m)
    Ys = OMIC.getBomicSI(n)

    omic = OMIC(R,Xs,Ys,Lambdas)
    omic.run()

    Z = omic.getRecoveredMatrix()

    print(Z)