import numpy 
from sklearn.decomposition import PCA


def kl_div(p, q):
    """
    Empiric KL-divergence between p and q.
    
    Args:
        p: np.ndarray of length N;
        q: np.ndarray of length N;
    Returns:
        A number, KL(P || q).
    """
    eps = 1e-10
    p_safe = np.copy(p)
    p_safe[p_safe < eps] = eps
    q_safe = np.copy(q)
    q_safe[q_safe < eps] = eps
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))

def js_div(p, q):
    """
    Jensen–Shannon divergence.
    Args:
        p: np.ndarray of shape (N,)
        q: np.ndarray of shape (N,)
    Returns:
        A number, JS(P || q).
    """
    m = (p + q) / 2.
    return (kl_div(p, m) + kl_div(q, m)) / 2.

def pca(x_real, x_fake):
    """
    Applies pca to the two given datasets.
    
    Args:
        x_real: np.ndarray of shape (n_real, n_features)
        x_fake: np.ndarray of shape (n_fake, n_features)
    
    Returns
        x_real_pca: np.ndarray of shape (n_real, 2)
        x_fake_pca: np.ndarray of shape (n_fake, 2)
    """
    pca = PCA(n_components=2)
    x_real_pca = pca.fit_transform(x_real)
    x_fake_pca = pca.transform(x_fake)
    return x_real_pca, x_fake_pca
