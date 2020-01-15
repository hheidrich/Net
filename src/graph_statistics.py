import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import networkx as nx
import powerlaw


def max_degree(A_in):
    """Compute max degree."""
    degrees = A_in.sum(axis=-1)
    return np.max(degrees)


def min_degree(A_in):
    """Compute min degree."""
    degrees = A_in.sum(axis=-1)
    return np.min(degrees)


def average_degree(A_in):
    """Compute average degree."""
    degrees = A_in.sum(axis=-1)
    return np.mean(degrees)


def LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """
    G = nx.from_scipy_sparse_matrix(A_in)
    return max([len(c) for c in nx.connected_components(G)])

def num_connected_components(A_in):
    """Compute the number of connected components."""
    G = nx.from_scipy_sparse_matrix(A_in)
    return len(list(nx.connected_components(G)))


def wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = np.array(A_in.sum(axis=-1))
    return 0.5 * np.dot(degrees.T, degrees-1).reshape([])


def claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = np.array(A_in.sum(axis=-1))
    return 1/6 * np.sum(degrees * (degrees-1) * (degrees-2))


def triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_scipy_sparse_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_squared = A_in @ A_in
    common_neighbors = sp.triu(A_squared, k=1).tocsr()
    num_common_neighbors = np.array(common_neighbors[common_neighbors.nonzero()]).reshape(-1)
    return np.dot(num_common_neighbors, num_common_neighbors-1) / 4


def power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = np.array(A_in.sum(axis=-1)).flatten()
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1), verbose=False).power_law.alpha


def gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """
    N = A_in.shape[0]
    degrees_sorted = np.sort(np.array(A_in.sum(axis=-1)).flatten())
    return 2 * np.dot(degrees_sorted, np.arange(1, N+1)) / (N * np.sum(degrees_sorted)) - (N+1) / N


def edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """
    N = A_in.shape[0]
    degrees = np.array(A_in.sum(axis=-1)).flatten()
    degrees /= degrees.sum()
    return -np.dot(np.log(degrees), degrees) / np.log(N)


def compute_cpl(A_in):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(A_in)
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def smallest_eigvals_of_LCC(A):
    """Computes few smallest eigenvalues of graph Laplacian, restricted to largest connected component."""
    G = nx.from_scipy_sparse_matrix(A)
    Gc = G.subgraph(max(nx.connected_components(G), key=len))
    L = nx.normalized_laplacian_matrix(Gc)
    vals, vecs = eigs(L, k=2, sigma=-0.0001)
    return np.real(vals)


def spectral_gap(A_in):
    """ Compute spectral gap."""
    eigvals = smallest_eigvals_of_LCC(A_in)
    return eigvals[1] - eigvals[0]


def assortativity(A_in):
    """Compute assortativity."""
    G = nx.from_scipy_sparse_matrix(A_in)
    return nx.degree_assortativity_coefficient(G)


def clustering_coefficient(A_in):
    """Compute clustering coefficient."""
    return 3 * triangle_count(A_in) / claw_count(A_in)


def compute_graph_statistics(A):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
          
    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """
    statistics = {}

    # Degree statistics
    statistics['d_max'] = max_degree(A)
    statistics['d_min'] = min_degree(A)
    statistics['d'] = average_degree(A)
    # largest connected component
    statistics['LCC'] = LCC(A)
    # wedge count
    statistics['wedge_count'] = wedge_count(A)
    # claw count
    statistics['claw_count'] = claw_count(A)
    # triangle count
    statistics['triangle_count'] = triangle_count(A)
    # Square count
    statistics['square_count'] = square_count(A)
    # power law exponent
    statistics['power_law_exp'] = power_law_alpha(A)
    # gini coefficient
    statistics['gini'] = gini(A)
    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = edge_distribution_entropy(A)
    # Assortativity
    statistics['assortativity'] = assortativity(A)
    # Clustering coefficient
    statistics['clustering_coefficient'] = clustering_coefficient(A)
    # Number of connected components
    statistics['n_components'] = num_connected_components(A)
    # Characteristic path length
    statistics['cpl'] = compute_cpl(A)
    # Spectral gap of largest connected component
    statistics['spectral_gap'] = spectral_gap(A)
    return statistics
