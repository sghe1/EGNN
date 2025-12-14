import os
import matplotlib.pyplot as plt
import numpy as np

def make_final_plots(save_dir, train_losses, val_losses, train_vel, val_vel, train_stress, val_stress, predictions, targets):
    os.makedirs(save_dir, exist_ok=True)

    # 1. LOSS CURVES
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Convergence')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # 2. COMPONENT LOSSES
    plt.figure(figsize=(10, 5))
    plt.plot(train_vel, label='Train Vel', linestyle='--')
    plt.plot(val_vel, label='Val Vel')
    plt.plot(train_stress, label='Train Stress', linestyle='--')
    plt.plot(val_stress, label='Val Stress')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.title('Velocity vs Stress Loss')
    plt.savefig(os.path.join(save_dir, 'component_losses.png'))
    plt.close()

    # 3. PARITY PLOTS (Pred vs True)
    # predictions/targets shape: [N, 4] -> [Vel(3), Stress(1)]
    # We sample random points to avoid clogging the plot
    indices = np.random.choice(len(predictions), size=min(len(predictions), 5000), replace=False)
    pred_sub = predictions[indices]
    targ_sub = targets[indices]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity Magnitude Parity
    pred_vel_mag = np.linalg.norm(pred_sub[:, :3], axis=1)
    targ_vel_mag = np.linalg.norm(targ_sub[:, :3], axis=1)
    
    axes[0].scatter(targ_vel_mag, pred_vel_mag, alpha=0.3, s=1)
    axes[0].plot([0, targ_vel_mag.max()], [0, targ_vel_mag.max()], 'r--')
    axes[0].set_xlabel('Ground Truth Speed')
    axes[0].set_ylabel('Predicted Speed')
    axes[0].set_title('Velocity Parity')

    # Stress Parity
    axes[1].scatter(targ_sub[:, 3], pred_sub[:, 3], alpha=0.3, s=1)
    axes[1].plot([targ_sub[:, 3].min(), targ_sub[:, 3].max()], 
                 [targ_sub[:, 3].min(), targ_sub[:, 3].max()], 'r--')
    axes[1].set_xlabel('Ground Truth Stress')
    axes[1].set_ylabel('Predicted Stress')
    axes[1].set_title('Stress Parity')

    plt.savefig(os.path.join(save_dir, 'predictions.png'))
    plt.close()
    print(f"Plots saved to {save_dir}")