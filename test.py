import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Atom:
    def __init__(self, atom_id, grid_pos):
        self.id = atom_id
        self.grid_pos = tuple(grid_pos)

class NA3DCompiler_DualRadius:
    def __init__(self, size=7, layers=6):
        self.size = size
        self.layers = layers
        self.grid = {} 
        self.center_range = range(2, 5) # 中心 3x3 是计算区
        
        # 物理坐标系映射 (核心升级：引入真实的微米级物理间距)
        self.spacing_xy = 4.0  # XY平面原子间距 4 um
        self.spacing_z = 12.0  # Z轴层间距 12 um
        
        # 双轨安全半径 (Dual-Radius Logic)
        self.r_route = 2.0  # 搬运避让半径 (允许穿针引线)
        self.r_gate = 8.0   # 里德堡门计算半径 (必须绝对空旷)

    def get_phys_pos(self, grid_pos):
        """将离散的网格坐标转换为真实的 3D 物理坐标 (微米)"""
        return np.array([grid_pos[0] * self.spacing_xy, 
                         grid_pos[1] * self.spacing_xy, 
                         grid_pos[2] * self.spacing_z], dtype=float)

    def add_atom(self, atom_id, grid_pos):
        self.grid[tuple(grid_pos)] = Atom(atom_id, grid_pos)

    def get_zone(self, x, y, z):
        if z == self.layers - 1: return "READOUT"
        if x in self.center_range and y in self.center_range: return "INTERACTION"
        return "STORAGE"

    # ==========================================
    # 模块 1：双半径物理检查
    # ==========================================
    def is_safe_for_routing(self, grid_pos, moving_atom_id):
        """检查物理位置是否可以穿过去 (使用 R_route = 2 um)"""
        x, y, z = grid_pos
        if not (0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.layers):
            return False
            
        phys_pos = self.get_phys_pos(grid_pos)
        for atom in self.grid.values():
            if atom.id == moving_atom_id: continue
            atom_phys = self.get_phys_pos(atom.grid_pos)
            if np.linalg.norm(phys_pos - atom_phys) < self.r_route:
                return False # 连缝隙都没有，会撞车
        return True

    def is_safe_for_gate(self, grid_pos, moving_atom_id):
        """检查物理位置是否足够空旷以执行量子门 (使用 R_gate = 8 um)"""
        phys_pos = self.get_phys_pos(grid_pos)
        for atom in self.grid.values():
            if atom.id == moving_atom_id: continue
            atom_phys = self.get_phys_pos(atom.grid_pos)
            if np.linalg.norm(phys_pos - atom_phys) < self.r_gate:
                return False # 空间不足以展开里德堡封锁球，引发串扰
        return True

    # ==========================================
    # 模块 2：离散网格寻路 (物理距离代价 A*)
    # ==========================================
    def a_star_search(self, start, end, atom_id):
        start, end = tuple(start), tuple(end)
        end_phys = self.get_phys_pos(end)
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        directions = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==0 and dy==0 and dz==0)]

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            current_phys = self.get_phys_pos(current)
            for dx, dy, dz in directions:
                neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)
                if self.is_safe_for_routing(neighbor, atom_id):
                    neighbor_phys = self.get_phys_pos(neighbor)
                    cost = np.linalg.norm(neighbor_phys - current_phys) # 物理距离作为代价
                    tentative_g = g_score[current] + cost
                    
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + np.linalg.norm(neighbor_phys - end_phys)
                        heapq.heappush(open_set, (f_score, neighbor))
        return None

    # ==========================================
    # 模块 3：动态备选路由 (Gate Clearance Aware)
    # ==========================================
    def find_best_interaction_site(self, start_pos, atom_id):
        valid_paths = []
        for z in range(self.layers - 1):
            for y in self.center_range:
                for x in self.center_range:
                    candidate_dest = (x, y, z)
                    if candidate_dest in self.grid and candidate_dest != start_pos:
                        continue
                    
                    # 核心判断：这个备用点做 Gate 安全吗？
                    if not self.is_safe_for_gate(candidate_dest, atom_id):
                        continue
                        
                    path = self.a_star_search(start_pos, candidate_dest, atom_id)
                    if path:
                        # 计算物理路径总长度
                        path_phys_len = sum(np.linalg.norm(self.get_phys_pos(path[i+1]) - self.get_phys_pos(path[i])) for i in range(len(path)-1))
                        valid_paths.append({'dest': candidate_dest, 'path': path, 'length': path_phys_len})
        
        if not valid_paths: return None, None
        best_option = sorted(valid_paths, key=lambda k: k['length'])[0]
        return best_option['path'], best_option['dest']

    # ==========================================
    # 模块 4：连续空间直线平滑 (Line-of-Sight)
    # ==========================================
    def point_to_line_segment_dist(self, p, a, b):
        ab = b - a; ap = p - a
        if np.dot(ab, ab) == 0: return np.linalg.norm(ap)
        t = max(0, min(1, np.dot(ap, ab) / np.dot(ab, ab)))
        return np.linalg.norm(p - (a + t * ab))

    def smooth_path(self, path, atom_id):
        if not path or len(path) <= 2: return path
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            for look_ahead in range(len(path) - 1, current_idx, -1):
                # 检查直线上的物理安全距离 (使用 R_route 允许穿梭)
                safe_line = True
                a_phys = self.get_phys_pos(path[current_idx])
                b_phys = self.get_phys_pos(path[look_ahead])
                
                for atom in self.grid.values():
                    if atom.id == atom_id: continue
                    atom_phys = self.get_phys_pos(atom.grid_pos)
                    if self.point_to_line_segment_dist(atom_phys, a_phys, b_phys) < self.r_route:
                        safe_line = False; break
                
                if safe_line:
                    smoothed.append(path[look_ahead])
                    current_idx = look_ahead
                    break
        return smoothed

    # ==========================================
    # 模块 5：物理可视化
    # ==========================================
    def visualize(self, smooth_path=None, actual_target=None):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(projection='3d')
        
        for atom in self.grid.values():
            if atom.id == "Q_target": continue
            phys_pos = self.get_phys_pos(atom.grid_pos)
            ax.scatter(*phys_pos, c='royalblue', s=80, alpha=0.6, edgecolors='w')

        if smooth_path:
            sp_phys = np.array([self.get_phys_pos(p) for p in smooth_path])
            ax.plot(sp_phys[:,0], sp_phys[:,1], sp_phys[:,2], c='gold', linewidth=3, marker='o', label='Weaving Path')
            ax.scatter(*sp_phys[0], c='green', s=200, marker='*', label='Start')
            
            # 绘制终点的 Rydberg 封锁球 (R_gate = 8 um)
            end_phys = sp_phys[-1]
            ax.scatter(*end_phys, c='red', s=200, marker='X', label=f'Gate Site {actual_target}')
            
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = end_phys[0] + self.r_gate * np.cos(u) * np.sin(v)
            y = end_phys[1] + self.r_gate * np.sin(u) * np.sin(v)
            z = end_phys[2] + self.r_gate * np.cos(v)
            ax.plot_wireframe(x, y, z, color="red", alpha=0.1, label='Rydberg Blockade Sphere')

        ax.set_title("3D NA Compiler: Dual-Radius Weaving & Gate Placement")
        ax.set_xlabel('X ($\mu m$)'); ax.set_ylabel('Y ($\mu m$)'); ax.set_zlabel('Z ($\mu m$)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        ax.set_box_aspect([self.size*self.spacing_xy, self.size*self.spacing_xy, self.layers*self.spacing_z])
        plt.show()

# ==========================================
# 运行演示场景：穿梭与计算检查
# ==========================================
compiler = NA3DCompiler_DualRadius()

start_pos = (0, 0, 0)
compiler.add_atom("Q_target", start_pos)
primary_target = (3, 3, 2)

print(f"🚀 编译器任务：将原子从 {start_pos} 移动到首选计算位 {primary_target}")

# 在目标周围放一圈原子，距离是 4 um (因为 spacing_xy = 4.0)
caged_count = 0
for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
    obs_pos = (primary_target[0]+dx, primary_target[1]+dy, primary_target[2])
    compiler.add_atom(f"Obs_{caged_count}", obs_pos)
    caged_count += 1

print(f"🚧 物理环境：目标周围 4 um 处存在 {caged_count} 个原子。")
print(f"   [规则] R_route (允许穿梭) = 2 um, R_gate (做门要求) = 8 um")

# 1. 检查物理穿梭是否可行？(能钻进去吗？)
can_route = compiler.is_safe_for_routing(primary_target, "Q_target")
print(f"🏃 穿梭检查：目标点物理上可达吗？ -> {'✅ 是的，缝隙够大，可以钻进去！' if can_route else '❌ 否'}")

# 2. 检查计算是否安全？(能在这里做门吗？)
can_gate = compiler.is_safe_for_gate(primary_target, "Q_target")
print(f"⚡ 计算检查：目标点适合展开里德堡球吗？ -> {'✅ 是的' if can_gate else '❌ 否，周围原子小于 8 um，会引发串扰！'}")

final_path = None
actual_target = primary_target

if can_route and can_gate:
    final_path = compiler.a_star_search(start_pos, primary_target, "Q_target")
else:
    print("\n🔍 触发动态备选路由 (寻找具备 8 um 净空的计算位)...")
    fallback_path, new_target = compiler.find_best_interaction_site(start_pos, "Q_target")
    
    if fallback_path:
        print(f"✅ 成功找到无串扰计算位！目标已重定向至: {new_target}")
        final_path = fallback_path
        actual_target = new_target

if final_path:
    print(f"📐 正在拉直路径并穿梭 (Line-of-Sight)...")
    smooth_path = compiler.smooth_path(final_path, "Q_target")
    compiler.visualize(smooth_path=smooth_path, actual_target=actual_target)