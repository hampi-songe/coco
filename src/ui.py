import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap
import json
import os
import torch
import warnings
import subprocess
import threading
import sys
import random
import time

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 样式常量 (亮色科技风格) ---
BG_COLOR = "#f8fafc"         # 极浅灰蓝色背景 (亮色)
CARD_BG = "#ffffff"          # 纯白卡片内容区
CARD_ACCENT = "#f1f5f9"      # 半透明灰色感背景
ACCENT_COLOR = "#2563eb"     # 科技蓝色
ACCENT_GLOW = "#3b82f6"      # 辅助蓝色
SECONDARY_ACCENT = "#6366f1" # 紫蓝色
TEXT_PRIMARY = "#0f172a"     # 深色文字
TEXT_DIM = "#64748b"         # 灰度文字
SHADOW_COLOR = "#e2e8f0"     # 浅灰色阴影

class RoundedCard(tk.Canvas):
    """具有圆角和阴影效果的自定义卡片容器 (亮色半透明感)"""
    def __init__(self, master, radius=20, shadow_offset=5, bg=CARD_BG, shadow_color=SHADOW_COLOR, **kwargs):
        # 获取 master 的背景色
        m_bg = master.cget("bg") if hasattr(master, "cget") else BG_COLOR
        if not m_bg: m_bg = BG_COLOR
        super().__init__(master, bg=m_bg, highlightthickness=0, **kwargs)
        self.radius = radius
        self.shadow_offset = shadow_offset
        self.card_bg = bg
        self.shadow_color = shadow_color
        self.inner_frame = tk.Frame(self, bg=bg)
        self.bind("<Configure>", self._draw)

    def _draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        r = self.radius
        s = self.shadow_offset
        
        # 绘制阴影
        for i in range(s, 0, -1):
            self._draw_round_rect(i, i, w-s+i, h-s+i, r, fill=self.shadow_color, outline="")
        
        # 绘制主卡片 (半透明灰色外框感)
        self._draw_round_rect(0, 0, w-s, h-s, r, fill=CARD_ACCENT, outline="#e2e8f0")
        
        # 放置内部内容框架 (纯白)
        margin = 2 # 留出一圈灰色外框，模拟半透明感
        self.create_window(margin, margin, window=self.inner_frame, anchor="nw", 
                          width=w-s-2*margin, height=h-s-2*margin)

    def _draw_round_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def get_frame(self):
        return self.inner_frame

class TechBackground(tk.Canvas):
    """动态科技感背景 (亮色版)"""
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=BG_COLOR, highlightthickness=0, **kwargs)
        self.particles = []
        self.num_particles = 40
        self.bind("<Configure>", self._init_particles)
        self.animate()

    def _init_particles(self, event=None):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10: return
        self.particles = []
        for _ in range(self.num_particles):
            self.particles.append({
                "x": random.randint(0, w),
                "y": random.randint(0, h),
                "vx": random.uniform(-0.4, 0.4),
                "vy": random.uniform(-0.4, 0.4),
                "r": random.randint(1, 3)
            })

    def animate(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w > 10 and h > 10:
            self.delete("all")
            
            # 绘制背景网格
            grid_size = 60
            for i in range(0, w, grid_size):
                self.create_line(i, 0, i, h, fill="#e2e8f0", width=1)
            for j in range(0, h, grid_size):
                self.create_line(0, j, w, j, fill="#e2e8f0", width=1)

            # 更新和绘制粒子
            for i, p in enumerate(self.particles):
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                
                if p["x"] < 0 or p["x"] > w: p["vx"] *= -1
                if p["y"] < 0 or p["y"] > h: p["vy"] *= -1
                
                self.create_oval(p["x"]-p["r"], p["y"]-p["r"], p["x"]+p["r"], p["y"]+p["r"], 
                                fill="#94a3b8", outline="")

                for j in range(i + 1, len(self.particles)):
                    p2 = self.particles[j]
                    dist = ((p["x"]-p2["x"])**2 + (p["y"]-p2["y"])**2)**0.5
                    if dist < 180:
                        alpha_val = int((1 - dist/180) * 40) + 220
                        hex_color = f"#{alpha_val:02x}{alpha_val:02x}{alpha_val:02x}"
                        self.create_line(p["x"], p["y"], p2["x"], p2["y"], fill="#cbd5e1", width=1)

        self.after(60, self.animate)

class AutonomousGamingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("自主博弈操作界面 - COCO Platform")
        self.root.geometry("1450x950")
        self.root.configure(bg=BG_COLOR)

        # 状态变量
        self.process = None
        self.is_running = False
        self.current_timestep = 0
        self.max_timesteps = 2050000
        
        # 字体
        self.font_title = ("SimSun", 22, "bold")
        self.font_main = ("SimSun", 11)
        self.font_bold = ("SimSun", 11, "bold")
        self.font_mono = ("Consolas", 10)

        self.maps = {
            "1o_2r_vs_4r": "1个监察者, 2个蟑螂 vs 4个蟑螂。这是一个非对称地图，需要不同单位类型之间的协调。",
            "1c3s5z": "1个巨像, 3个追猎者, 5个狂热者 vs 1个巨像, 3个追猎者, 5个狂热者。这是一个具有混合近战和远程单位的异构地图。",
            "MMM": "机枪兵,掠夺者,医疗运输机 vs 机枪兵,掠夺者,医疗运输机。经典的 Terran 生物组合场景。",
            "3s_vs_5z": "3个追猎者 vs 5个狂热者。专注于远程单位风筝近战单位的能力。",
            "MMM2": "机枪兵,掠夺者,医疗运输机 vs 机枪兵,掠夺者,医疗运输机。MMM地图的更大版本，拥有更多单位。"
        }
        
        self.algorithms = ["COCO"]
        self.platforms = ["SMAC"]

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        # 1. 动态背景
        self.bg_canvas = TechBackground(self.root)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # 2. 顶部标题栏
        header_card = RoundedCard(self.root, radius=15, shadow_offset=3, height=85)
        header_card.pack(fill=tk.X, padx=20, pady=15)
        header_frame = header_card.get_frame()
        
        tk.Label(header_frame, text="自主博弈操作系统 (COCO Platform)", font=self.font_title, 
                 fg=ACCENT_COLOR, bg=CARD_BG).pack(pady=12)

        # 3. 主容器
        main_container = tk.Frame(self.root, bg=BG_COLOR)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        # --- 左侧控制面板 ---
        left_card = RoundedCard(main_container, radius=20, shadow_offset=4, width=390)
        left_card.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_card.pack_propagate(False)
        left_frame = left_card.get_frame()

        # 模块 1: 配置选择
        self._create_section_label(left_frame, "⚙️ 系统配置").pack(anchor=tk.W, pady=(5, 10))
        
        self._create_field_label(left_frame, "博弈平台").pack(anchor=tk.W)
        self.platform_var = tk.StringVar(value=self.platforms[0])
        platform_cb = ttk.Combobox(left_frame, textvariable=self.platform_var, values=self.platforms, state="readonly")
        platform_cb.pack(fill=tk.X, pady=(0, 15))

        self._create_field_label(left_frame, "实验场景").pack(anchor=tk.W)
        self.map_var = tk.StringVar(value=list(self.maps.keys())[0])
        self.map_var.trace("w", self.on_map_change)
        map_cb = ttk.Combobox(left_frame, textvariable=self.map_var, values=list(self.maps.keys()), state="readonly")
        map_cb.pack(fill=tk.X, pady=(0, 10))

        self.map_desc_text = tk.Text(left_frame, height=5, font=self.font_main, bg=CARD_ACCENT, fg=TEXT_DIM, 
                                    relief=tk.FLAT, padx=12, pady=10)
        self.map_desc_text.pack(fill=tk.X, pady=(0, 15))
        self.update_map_desc()

        self._create_field_label(left_frame, "训练算法").pack(anchor=tk.W)
        self.algo_var = tk.StringVar(value=self.algorithms[0])
        algo_cb = ttk.Combobox(left_frame, textvariable=self.algo_var, values=self.algorithms, state="readonly")
        algo_cb.pack(fill=tk.X, pady=(0, 20))

        self.run_btn = tk.Button(left_frame, text="▶ 开始运行实验", font=self.font_bold, 
                                bg=ACCENT_COLOR, fg="white", activebackground=ACCENT_GLOW, 
                                relief=tk.FLAT, pady=12, command=self.toggle_experiment, cursor="hand2")
        self.run_btn.pack(fill=tk.X, pady=(0, 20))

        # 模块 2: 状态监控
        self._create_section_label(left_frame, "📊 实时监控").pack(anchor=tk.W, pady=(10, 5))
        self.timestep_label = tk.Label(left_frame, text="当前步数: 0 / 2,050,000", font=self.font_bold, 
                                      fg=ACCENT_COLOR, bg=CARD_BG)
        self.timestep_label.pack(anchor=tk.W, pady=(0, 10))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(left_frame, variable=self.progress_var, maximum=2050000)
        self.progress_bar.pack(fill=tk.X, pady=(0, 15))

        self._create_field_label(left_frame, "控制台输出").pack(anchor=tk.W)
        self.log_area = scrolledtext.ScrolledText(left_frame, height=12, font=self.font_mono, 
                                                bg=CARD_ACCENT, fg="#1e293b", relief=tk.FLAT)
        self.log_area.pack(fill=tk.BOTH, expand=True)

        # --- 右侧内容区域 ---
        right_container = tk.Frame(main_container, bg=BG_COLOR)
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 上方：胜率曲线
        curve_card = RoundedCard(right_container, radius=20, shadow_offset=4)
        curve_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        curve_frame = curve_card.get_frame()
        
        tk.Label(curve_frame, text="📈 实验结果曲线 (Win Rate)", font=self.font_bold, 
                 fg=ACCENT_COLOR, bg=CARD_BG).pack(pady=(5, 5))
        
        self.fig_curve, self.ax_curve = plt.subplots(figsize=(8, 4), dpi=100)
        self.fig_curve.patch.set_facecolor(CARD_BG)
        self.ax_curve.set_facecolor(CARD_ACCENT)
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, master=curve_frame)
        self.canvas_curve.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 下方：t-SNE 可视化
        tsne_card = RoundedCard(right_container, radius=20, shadow_offset=4)
        tsne_card.pack(fill=tk.BOTH, expand=True)
        tsne_frame = tsne_card.get_frame()
        
        tk.Label(tsne_frame, text="🔮 消息空间可视化 (t-SNE)", font=self.font_bold, 
                 fg=ACCENT_COLOR, bg=CARD_BG).pack(pady=(5, 5))
        
        self.fig_tsne, self.ax_tsne = plt.subplots(figsize=(6, 6), dpi=100)
        self.fig_tsne.patch.set_facecolor(CARD_BG)
        self.ax_tsne.set_facecolor(CARD_ACCENT)
        self.canvas_tsne = FigureCanvasTkAgg(self.fig_tsne, master=tsne_frame)
        self.canvas_tsne.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 底部状态栏
        self.status_bar = tk.Label(self.root, text="系统就绪 | 正在监控 SMAC 平台...", bd=0, anchor=tk.W,
                                  bg="#e2e8f0", fg=TEXT_DIM, font=("SimSun", 9), padx=20, pady=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_section_label(self, master, text):
        return tk.Label(master, text=text, font=("SimSun", 14, "bold"), fg=SECONDARY_ACCENT, bg=CARD_BG)

    def _create_field_label(self, master, text):
        return tk.Label(master, text=text, font=self.font_bold, fg=TEXT_PRIMARY, bg=CARD_BG)

    def on_map_change(self, *args):
        self.status_bar.config(text=f"正在切换场景: {self.map_var.get()}...")
        self.update_map_desc()
        self.load_data()
        self.update_plots()
        self.status_bar.config(text=f"场景 {self.map_var.get()} 加载完成")

    def update_plots(self):
        self.plot_curve()
        self.plot_tsne()

    def update_map_desc(self):
        self.map_desc_text.config(state=tk.NORMAL)
        self.map_desc_text.delete(1.0, tk.END)
        self.map_desc_text.insert(tk.END, self.maps.get(self.map_var.get(), ""))
        self.map_desc_text.config(state=tk.DISABLED)

    def toggle_experiment(self):
        if not self.is_running:
            self.start_experiment()
        else:
            self.stop_experiment()

    def start_experiment(self):
        map_name = self.map_var.get()
        algo_name = self.algo_var.get().lower()
        python_exe = "A:/Conda/envs/smac/python.exe"
        main_script = os.path.join(os.path.dirname(__file__), "main.py")
        cmd = [python_exe, main_script, "--env-config=sc2", f"--config={algo_name}", "with", f"env_args.map_name={map_name}"]
        self.log_area.delete(1.0, tk.END)
        self.log_area.insert(tk.END, f"🚀 正在启动实验: {map_name}...\n")
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
                cwd=os.path.dirname(main_script)
            )
            self.is_running = True
            self.run_btn.config(text="⏹ 停止实验", bg="#ef4444")
            threading.Thread(target=self.read_logs, daemon=True).start()
            self.monitor_progress()
        except Exception as e:
            messagebox.showerror("启动失败", f"无法启动实验: {str(e)}")

    def stop_experiment(self):
        if self.process:
            self.process.terminate()
            self.log_area.insert(tk.END, "\n--- 实验已手动停止 ---\n")
            self.is_running = False
            self.run_btn.config(text="▶ 开始运行实验", bg=ACCENT_COLOR)

    def read_logs(self):
        while self.is_running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if line:
                self.root.after(0, self.update_log_area, line)
        if self.process:
            self.root.after(0, self.finish_experiment)

    def update_log_area(self, line):
        self.log_area.insert(tk.END, line)
        self.log_area.see(tk.END)

    def finish_experiment(self):
        self.is_running = False
        self.run_btn.config(text="▶ 开始运行实验", bg=ACCENT_COLOR)
        self.load_data()
        self.update_plots()

    def monitor_progress(self):
        if not self.is_running: return
        if not hasattr(self, 'last_plot_timestep'): self.last_plot_timestep = -1
        self.load_data()
        self.timestep_label.config(text=f"当前步数: {self.current_timestep:,} / 2,050,000")
        self.progress_var.set(self.current_timestep)
        if self.current_timestep - self.last_plot_timestep >= 10000:
            self.update_plots()
            self.last_plot_timestep = self.current_timestep
        self.root.after(10000, self.monitor_progress)

    def load_data(self):
        try:
            results_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
            sacred_dir = os.path.join(results_root, "sacred")
            if os.path.exists(sacred_dir):
                runs = [d for d in os.listdir(sacred_dir) if d.isdigit()]
                if runs:
                    latest_run = sorted(runs, key=int)[-1]
                    info_path = os.path.join(sacred_dir, latest_run, "info.json")
                    if os.path.exists(info_path):
                        with open(info_path, "r") as f:
                            self.data_info = json.load(f)
                            if "battle_won_mean_T" in self.data_info:
                                self.current_timestep = max(self.data_info["battle_won_mean_T"])
            msg_root = os.path.join(results_root, "messages")
            if os.path.exists(msg_root):
                msg_dirs = os.listdir(msg_root)
                if msg_dirs:
                    self.latest_msg_path = max([os.path.join(msg_root, d) for d in msg_dirs], key=os.path.getmtime)
        except Exception as e:
            print(f"Error loading data: {e}")

    def plot_curve(self):
        self.ax_curve.clear()
        self.ax_curve.set_facecolor(CARD_ACCENT)
        if hasattr(self, 'data_info') and self.data_info and "battle_won_mean" in self.data_info:
            y = self.data_info["battle_won_mean"]
            x = self.data_info.get("battle_won_mean_T", list(range(len(y))))
            self.ax_curve.plot(x, y, color=ACCENT_COLOR, linewidth=2.5, label='COCO')
            self.ax_curve.fill_between(x, y, color=ACCENT_COLOR, alpha=0.1)
            self.ax_curve.legend(facecolor=CARD_BG, edgecolor=SHADOW_COLOR, labelcolor=TEXT_PRIMARY)
            self.ax_curve.set_xlabel("时间步", color=TEXT_DIM, fontproperties="SimSun")
            self.ax_curve.set_ylabel("平均测试胜率", color=TEXT_DIM, fontproperties="SimSun")
            self.ax_curve.grid(True, linestyle='-', alpha=0.1, color="#cbd5e1")
            self.ax_curve.tick_params(colors=TEXT_DIM, labelsize=9)
            self.ax_curve.set_xlim(0, 2050000)
            self.ax_curve.set_ylim(0, 1.1)
            for spine in self.ax_curve.spines.values(): spine.set_color("#e2e8f0")
        else:
            self.ax_curve.text(0.5, 0.5, "等待实验数据...", color=TEXT_DIM, ha='center', fontproperties="SimSun")
        self.fig_curve.tight_layout()
        self.canvas_curve.draw()

    def plot_tsne(self):
        self.ax_tsne.clear()
        self.ax_tsne.set_facecolor(CARD_ACCENT)
        msgs = None
        if hasattr(self, 'latest_msg_path') and self.latest_msg_path:
            try:
                msg_files = [f for f in os.listdir(self.latest_msg_path) if f.endswith('.pt')]
                if msg_files:
                    ts_files = sorted([int(f.replace('.pt', '')) for f in msg_files])
                    best_ts = max([ts for ts in ts_files if ts <= self.current_timestep] or [ts_files[0]])
                    msgs = torch.load(os.path.join(self.latest_msg_path, f"{best_ts}.pt"), map_location='cpu')
            except: pass

        if msgs is not None:
            steps, n_agents, _, msg_dim = msgs.shape
            data = msgs.mean(dim=2).permute(1, 0, 2).numpy() # [N, T, D]
            n_agents, max_step, msg_dim = data.shape
            reshaped_data = np.reshape(data.transpose(1, 0, 2), (n_agents * max_step, msg_dim))
            try:
                tsne = TSNE(n_components=2, early_exaggeration=12, perplexity=min(5, reshaped_data.shape[0]-1), init='pca', random_state=42)
                embedded = tsne.fit_transform(reshaped_data)
                colormaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
                for i in range(n_agents):
                    cmap = get_cmap(colormaps[i % len(colormaps)])
                    for t in range(max_step):
                        self.ax_tsne.scatter(embedded[i*max_step+t, 0], embedded[i*max_step+t, 1], 
                                           color=cmap(0.4 + 0.6*t/max_step), s=70, edgecolors='white', linewidth=0.3)
            except: pass
        else:
            self.ax_tsne.text(0.5, 0.5, "等待消息空间数据...", color=TEXT_DIM, ha='center', fontproperties="SimSun")
        self.ax_tsne.set_xticks([])
        self.ax_tsne.set_yticks([])
        for spine in self.ax_tsne.spines.values(): spine.set_visible(False)
        self.fig_tsne.tight_layout()
        self.canvas_tsne.draw()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TCombobox", fieldbackground=CARD_BG, background=ACCENT_COLOR, foreground=TEXT_PRIMARY, arrowcolor=TEXT_PRIMARY)
    style.configure("TProgressbar", thickness=10, troughcolor=CARD_ACCENT, background=ACCENT_COLOR)
    app = AutonomousGamingUI(root)
    root.mainloop()
