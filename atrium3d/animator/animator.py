import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_animation(compiler, save_path: str = "routing_animation.gif", fps: int = 15):
    """
    生成并保存原子的 3D 飞行穿梭动画。
    
    :param compiler: Atrium3D 实例对象，提供背景绘制与路由数据
    :param save_path: 动画保存路径
    :param fps: 动画帧率
    """
    frames_data = compiler.results_code.get("routing_frames")
    if not frames_data:
        raise ValueError("[Error] No routing frames found. Make sure to run solve(do_routing=True) first.")

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(projection="3d")

    # 绘制静态架构背景 (利用 Compiler 内置的方法)
    compiler._plot_architecture_background(ax, alpha_storage=0.1, alpha_interaction=0.1, alpha_readout=0.1)

    # 初始化动态原子的散点图
    scatter = ax.scatter([], [], [], c="#d62728", s=100, edgecolors="white", linewidths=1.0, alpha=0.9)
    
    # 预先生成所有 qubit 的 ID 文本以便在动画中移动
    texts = [ax.text(0, 0, 0, str(q), fontsize=8, color="black", weight='bold') 
             for q in range(compiler.results_code['n_qubits'])]

    # 设置坐标轴标签和物理比例
    ax.set_xlabel(r"X ($\mu m$)")
    ax.set_ylabel(r"Y ($\mu m$)")
    ax.set_zlabel(r"Z ($\mu m$)")
    ax.set_box_aspect([
        compiler.size * compiler.spacing_xy, 
        compiler.size * compiler.spacing_xy, 
        compiler.layers * compiler.spacing_z
    ])

    def update(frame_idx):
        current_positions = frames_data[frame_idx]
        xs, ys, zs = [], [], []
        
        for q in range(compiler.results_code['n_qubits']):
            pos = current_positions[str(q)]
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
            
            # 动态更新文本标签位置 (略微偏移以防遮挡)
            texts[q].set_position((pos[0] + 0.5, pos[1] + 0.5))
            texts[q].set_3d_properties(pos[2] + 0.5, 'z')

        # 在 Matplotlib 中更新 3D scatter 的坐标
        scatter._offsets3d = (xs, ys, zs)
        
        # 显示进度标题
        ax.set_title(f"3D Atom Shuttling: {compiler.benchmark}\nFrame {frame_idx}/{len(frames_data)}")
        
        return scatter, *texts

    print(f"[INFO] Rendering {len(frames_data)} frames... This may take a moment.")
    
    # 使用 pillow 库保存 GIF
    anim = FuncAnimation(fig, update, frames=len(frames_data), interval=1000/fps, blit=False)
    anim.save(save_path, writer='pillow', fps=fps, dpi=120)
    
    plt.close(fig)
    print(f"[INFO] ✨ Animation successfully saved to: {save_path}")