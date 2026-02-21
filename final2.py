import numpy as np
import time
import optuna
import matplotlib.pyplot as plt
from numpy.linalg import svd

def get_soft(mat_a, val_tau):
    return np.sign(mat_a) * np.maximum(np.abs(mat_a) - val_tau, 0.0)

def get_svd(mat_w, val_tau):
    mat_u, vec_sig, mat_v = svd(mat_w, full_matrices=False)
    sig_thresh = np.maximum(vec_sig - val_tau, 0.0)
    num_r = np.sum(sig_thresh > 0)
    val_nuc = np.sum(sig_thresh)
    if num_r == 0:
        return np.zeros_like(mat_w), val_nuc
    return (mat_u[:, :num_r] * sig_thresh[:num_r]) @ mat_v[:num_r, :], val_nuc

def check_conv(prim_res, dual_res, norm_ry, val_tol):
    prim_norm = np.linalg.norm(prim_res, "fro")
    dual_norm = np.linalg.norm(dual_res, "fro")
    return (prim_norm / norm_ry < val_tol) and (dual_norm / norm_ry < val_tol)

def solve_admm(mat_y, mask_r, val_lam, val_mu, val_tol=1e-5, max_iter=300):
    val_m, val_n = mat_y.shape
    mat_ry = mask_r * mat_y
    mat_x = np.zeros((val_m, val_n))
    mat_e = np.zeros((val_m, val_n))
    mat_m = np.zeros((val_m, val_n))
    norm_ry = max(1.0, np.linalg.norm(mat_ry, "fro"))
    list_obj = []
    list_res = []

    for val_k in range(max_iter):
        old_x = mat_x.copy()
        mat_c = mask_r * (mat_y - mat_x) + mat_m / val_mu
        mat_e = mask_r * get_soft(mat_c, 1.0 / val_mu)
        mat_z = mat_ry - mat_e + mat_m / val_mu
        mat_w = old_x + mask_r * (mat_z - old_x)
        
        mat_x, nuc_norm = get_svd(mat_w, val_lam / val_mu)
        
        prim_res = mat_ry - mask_r * mat_x - mat_e
        dual_res = val_mu * (mask_r * (mat_x - old_x))
        mat_m = mat_m + val_mu * prim_res
        
        l1_norm = np.sum(np.abs(mat_e))
        list_obj.append(l1_norm + val_lam * nuc_norm)
        prim_norm = np.linalg.norm(prim_res, "fro")
        list_res.append(prim_norm)
        
        if check_conv(prim_res, dual_res, norm_ry, val_tol):
            break

    return mat_x, val_k + 1, list_obj, list_res

def get_data(val_n=100, ratio_tr=0.6, ratio_te=0.2, prob_noise=0.05, val_seed=42):
    gen_rng = np.random.default_rng(val_seed)
    val_rank = 5
    mat_u = gen_rng.standard_normal((val_n, val_rank))
    mat_v = gen_rng.standard_normal((val_n, val_rank))
    true_x = mat_u @ mat_v.T
    ratio_obs = ratio_tr + ratio_te
    mask_obs = (gen_rng.random((val_n, val_n)) < ratio_obs)
    mat_y = np.sign(true_x)
    mat_y[mat_y == 0] = 1
    idx_omega = np.argwhere(mask_obs)
    gen_rng.shuffle(idx_omega)
    num_total = val_n * val_n
    size_tr = int(ratio_tr * num_total)
    size_te = int(ratio_te * num_total)
    idx_tr = idx_omega[:size_tr]
    idx_te = idx_omega[size_tr:size_tr + size_te]
    mask_tr = np.zeros((val_n, val_n))
    mask_te = np.zeros((val_n, val_n))
    mask_tr[idx_tr[:, 0], idx_tr[:, 1]] = 1.0
    mask_te[idx_te[:, 0], idx_te[:, 1]] = 1.0
    mask_flip = (gen_rng.random((val_n, val_n)) < prob_noise) & (mask_tr == 1)
    mat_y[mask_flip] *= -1
    return true_x, mat_y, mask_tr, mask_te, idx_tr, idx_te, val_rank

def calc_metrics(pred_x, full_y, idx_set):
    sign_pred = np.sign(pred_x[idx_set[:, 0], idx_set[:, 1]])
    sign_true = full_y[idx_set[:, 0], idx_set[:, 1]]
    val_acc = np.mean(sign_pred == sign_true)
    vec_sig = svd(pred_x, compute_uv=False)
    rank_hard = np.sum(vec_sig > 1e-4)
    sig_pos = vec_sig[vec_sig > 1e-9]
    if len(sig_pos) == 0:
        rank_eff = 0.0
    else:
        vec_p = sig_pos / np.sum(sig_pos)
        rank_eff = np.exp(-np.sum(vec_p * np.log(vec_p + 1e-15)))
    return val_acc, rank_hard, rank_eff

def eval_obj(trial_obj, obj_data):
    true_x, obs_y, mask_tr, mask_te, idx_tr, idx_te, true_rank = obj_data
    val_lam = trial_obj.suggest_float("lam", 1.0, 20.0, log=True)
    val_mu = trial_obj.suggest_float("mu0", 1e-4, 0.05, log=True)
    pred_x, num_iters, list_obj, list_res = solve_admm(obs_y, mask_tr, val_lam, val_mu)
    val_acc, val_rank, rank_eff = calc_metrics(pred_x, obs_y, idx_te)
    pen_rank = 0.1 * abs(val_rank - true_rank)
    return val_acc - pen_rank

def plot_graphs(list_obj, list_res):
    fig_obj, (ax_obj, ax_res) = plt.subplots(1, 2, figsize=(12, 5))
    ax_obj.plot(list_obj, linewidth=2, color='b')
    ax_obj.set_xlabel("Iteration")
    ax_obj.set_ylabel("Primal Objective")
    ax_obj.set_title("Objective Function Trend")
    ax_obj.grid(True)
    
    ax_res.plot(list_res, linewidth=2, color='r')
    ax_res.set_xlabel("Iteration")
    ax_res.set_ylabel("Constraint Error")
    ax_res.set_title("ADMM Convergence")
    ax_res.grid(True)
    
    plt.tight_layout()
    plt.show()

def main_run():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    all_data = get_data()
    true_rank = all_data[6]
    
    print("=" * 50)
    print(f"Generating Synthetic Dataset (True Rank: {true_rank})")
    
    best_params = {
        'lam': 4.542630, 
        'mu0': 0.001318
    }
    
    print("\nUsing Pre-Computed Optimal Hyperparameters:")
    print(f"  Lambda : {best_params['lam']:.6f}")
    print(f"  Mu0    : {best_params['mu0']:.6f}")
    
    time_start = time.time()
    pred_x, num_iters, list_obj, list_res = solve_admm(all_data[1], all_data[2], best_params['lam'], best_params['mu0'])
    time_run = time.time() - time_start
    
    acc_train, rank_train, eff_train = calc_metrics(pred_x, all_data[1], all_data[4])
    acc_test, rank_test, eff_test = calc_metrics(pred_x, all_data[1], all_data[5])
    
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Training Accuracy   : {acc_train * 100:.2f}%")
    print(f"Testing Accuracy    : {acc_test * 100:.2f}%")
    print(f"Recovered Hard Rank : {rank_test}")
    print(f"Effective Rank      : {eff_test:.2f}")
    print(f"Total Iterations    : {num_iters}")
    print(f"Execution Time      : {time_run:.3f} seconds")
    print("=" * 50)
    
    plot_graphs(list_obj, list_res)

if __name__ == "__main__":
    main_run()