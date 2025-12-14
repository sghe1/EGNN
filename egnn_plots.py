import os
import matplotlib.pyplot as plt
import numpy as np

def make_final_plots(save_dir, train_losses, val_losses, 
                     train_vel_losses=None, val_vel_losses=None, 
                     train_stress_losses=None, val_stress_losses=None, 
                     predictions=None, targets=None):
    
    os.makedirs(save_dir, exist_ok=True)

    # 1. Main Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title('Training Convergence')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # 2. Components
    if train_vel_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(train_vel_losses, label='Train Vel', linestyle='--')
        plt.plot(val_vel_losses, label='Val Vel')
        plt.plot(train_stress_losses, label='Train Stress', linestyle='--')
        plt.plot(val_stress_losses, label='Val Stress')
        plt.yscale('log')
        plt.xlabel('Epoch'); plt.ylabel('MSE')
        plt.legend(); plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title('Velocity vs Stress Loss')
        plt.savefig(os.path.join(save_dir, 'component_losses.png'))
        plt.close()

    # 3. Parity (Pred vs True)
    if predictions is not None:
        # Sample points if too many
        if len(predictions) > 5000:
            idx = np.random.choice(len(predictions), 5000, replace=False)
            pred = predictions[idx]
            targ = targets[idx]
        else:
            pred, targ = predictions, targets

        # Velocity Magnitude
        pred_v = np.linalg.norm(pred[:, :3], axis=1)
        targ_v = np.linalg.norm(targ[:, :3], axis=1)
        
        # Stress
        pred_s = pred[:, 3]
        targ_s = targ[:, 3]

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Vel Plot
        ax[0].scatter(targ_v, pred_v, alpha=0.3, s=2)
        m = max(targ_v.max(), pred_v.max())
        ax[0].plot([0, m], [0, m], 'r--')
        ax[0].set_title("Velocity Parity")
        ax[0].set_xlabel("Ground Truth"); ax[0].set_ylabel("Prediction")

        # Stress Plot
        ax[1].scatter(targ_s, pred_s, alpha=0.3, s=2)
        m_min, m_max = min(targ_s.min(), pred_s.min()), max(targ_s.max(), pred_s.max())
        ax[1].plot([m_min, m_max], [m_min, m_max], 'r--')
        ax[1].set_title("Stress Parity")
        ax[1].set_xlabel("Ground Truth"); ax[1].set_ylabel("Prediction")

        plt.savefig(os.path.join(save_dir, 'predictions.png'))
        plt.close()