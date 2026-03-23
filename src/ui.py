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

# --- 样式常量 (亮色科技风格 - 极致半透明) ---
BG_COLOR = "#f8fafc"         # 极浅灰蓝色背景
CARD_BORDER = "#cbd5e1"      # 卡片边框颜色
ACCENT_COLOR = "#2563eb"     # 科技蓝色
ACCENT_GLOW = "#3b82f6"      # 辅助蓝色
SECONDARY_ACCENT = "#6366f1" # 紫蓝色
TEXT_PRIMARY = "#0f172a"     # 深色文字
TEXT_DIM = "#64748b"         # 灰度文字
SHADOW_COLOR = "#ffffff"     # 浅灰色阴影

class TechBackground(tk.Canvas):
    """动态科技感背景 + 真正半透明卡片容器"""
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=BG_COLOR, highlightthickness=0, **kwargs)
        self.particles = []
        self.cards = [] 
        self.texts = [] # 存储需要在画布上直接绘制的文字
        self.num_particles = 60
        self.bind("<Configure>", self._on_resize)
        self.animate()

    def _on_resize(self, event=None):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10: return
        self._init_particles(w, h)

    def _init_particles(self, w, h):
        self.particles = []
        for _ in range(self.num_particles):
            self.particles.append({
                "x": random.randint(0, w),
                "y": random.randint(0, h),
                "vx": random.uniform(-0.6, 0.6),
                "vy": random.uniform(-0.6, 0.6),
                "r": random.randint(2, 5) # 调大点的初始半径
            })

    def add_card_shape(self, x, y, w, h, r=20, s=5):
        self.cards.append({"x": x, "y": y, "w": w, "h": h, "r": r, "s": s})

    def add_canvas_text(self, x, y, text, font, fill, anchor="nw"):
        self.texts.append({"x": x, "y": y, "text": text, "font": font, "fill": fill, "anchor": anchor})

    def _draw_round_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def animate(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w > 10 and h > 10:
            self.delete("bg_anim")
            
            # 1. 绘制背景网格
            grid_size = 60
            for i in range(0, w, grid_size):
                self.create_line(i, 0, i, h, fill="#f1f5f9", width=1, tags="bg_anim")
            for j in range(0, h, grid_size):
                self.create_line(0, j, w, j, fill="#f1f5f9", width=1, tags="bg_anim")

            # 2. 绘制粒子 (最底层)
            for i, p in enumerate(self.particles):
                p["x"] += p["vx"]; p["y"] += p["vy"]
                if p["x"] < 0 or p["x"] > w: p["vx"] *= -1
                if p["y"] < 0 or p["y"] > h: p["vy"] *= -1
                self.create_oval(p["x"]-p["r"], p["y"]-p["r"], p["x"]+p["r"], p["y"]+p["r"], fill="#cbd5e1", outline="", tags="bg_anim")
                for j in range(i + 1, len(self.particles)):
                    p2 = self.particles[j]
                    dist = ((p["x"]-p2["x"])**2 + (p["y"]-p2["y"])**2)**0.5
                    if dist < 150:
                        self.create_line(p["x"], p["y"], p2["x"], p2["y"], fill="#e2e8f0", width=2, tags="bg_anim")

            # 3. 绘制卡片 (半透明层)
            for card in self.cards:
                cx, cy, cw, ch, cr, cs = card["x"], card["y"], card["w"], card["h"], card["r"], card["s"]
                # 为阴影也添加半透明效果
                self._draw_round_rect(cx+cs, cy+cs, cx+cw+cs, cy+ch+cs, cr, fill=SHADOW_COLOR, outline="", stipple="gray12", tags="bg_anim")
                # 使用更轻的 stipple 模式 (gray25 或 gray12) 增加透明度
                self._draw_round_rect(cx, cy, cx+cw, cy+ch, cr, fill="white", outline=CARD_BORDER, stipple="gray25", tags="bg_anim")

            # 4. 绘制所有标签文字 (真正透明)
            for t in self.texts:
                self.create_text(t["x"], t["y"], text=t["text"], font=t["font"], fill=t["fill"], anchor=t["anchor"], tags="bg_anim")

            self.tag_lower("bg_anim")

        self.after(50, self.animate)

class AutonomousGamingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("自主博弈操作界面 - COCO Platform")
        self.root.geometry("1450x950")
        self.root.configure(bg=BG_COLOR)

        self.process = None
        self.is_running = False
        self.current_timestep = 0
        self.max_timesteps = 2050000
        
        # 字体
        self.font_title = ("Microsoft YaHei", 24, "bold")
        self.font_sec = ("Microsoft YaHei", 14, "bold")
        self.font_main = ("Microsoft YaHei", 12, "bold")
        self.font_bold = ("Microsoft YaHei", 12, "bold")
        self.font_mono = ("Consolas", 10, "bold")

        self.maps = {
            "1o_2r_vs_4r": "1个监察者, 2个蟑螂 vs 4个蟑螂。这是一个非对称地图，需要不同单位类型之间的协调。",
            "1c3s5z": "1个巨像, 3个追猎者, 5个狂热者 vs 1个巨像, 3个追猎者, 5个狂热者。这是一个具有混合近战和远程单位的异构地图。",
            "MMM": "机枪兵,掠夺者,医疗运输机 vs 机枪兵,掠夺者,医疗运输机。经典的 Terran 生物组合场景。",
            "3s_vs_5z": "3个追猎者 vs 5个狂热者。专注于远程单位风筝近战单位的能力。",
            "MMM2": "机枪兵,掠夺者,医疗运输机 vs 机枪兵,掠夺者,医疗运输机。MMM地图更大版本，拥有更多单位。"
        }
        self.algorithms = ["COCO"]; self.platforms = ["SMAC"]

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        self.bg_canvas = TechBackground(self.root)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)

        PAD = 20; LEFT_W = 380; HEADER_H = 80

        # --- 顶部标题 ---
        self.bg_canvas.add_card_shape(PAD, PAD, 1450-2*PAD, HEADER_H, r=15)
        self.bg_canvas.add_canvas_text(1450/2, PAD + HEADER_H/2, "自主博弈操作系统 (COCO Platform)", self.font_title, ACCENT_COLOR, anchor="center")

        # --- 左侧控制面板 ---
        LEFT_Y = PAD + HEADER_H + PAD
        self.bg_canvas.add_card_shape(PAD, LEFT_Y, LEFT_W, 950-LEFT_Y-PAD-40)
        
        # 使用 create_text 绘制所有静态标签
        self.bg_canvas.add_canvas_text(PAD+20, LEFT_Y+20, "⚙️ 系统配置", self.font_sec, "#334155")
        self.bg_canvas.add_canvas_text(PAD+20, LEFT_Y+60, "博弈平台", self.font_main, "#1e293b")
        self.bg_canvas.add_canvas_text(PAD+20, LEFT_Y+130, "实验场景", self.font_main, "#1e293b")
        self.bg_canvas.add_canvas_text(PAD+20, LEFT_Y+350, "训练算法", self.font_main, "#1e293b")
        self.bg_canvas.add_canvas_text(PAD+20, LEFT_Y+480, "📊 实时监控", self.font_sec, "#334155")
        self.bg_canvas.add_canvas_text(PAD+20, LEFT_Y+630, "控制台输出", self.font_main, "#1e293b")

        # 放置交互式组件 (Combobox, Button, Text, Progress)
        self.platform_var = tk.StringVar(value=self.platforms[0])
        platform_cb = ttk.Combobox(self.root, textvariable=self.platform_var, values=self.platforms, state="readonly")
        self.bg_canvas.create_window(PAD+20, LEFT_Y+85, window=platform_cb, anchor="nw", width=LEFT_W-40)

        self.map_var = tk.StringVar(value=list(self.maps.keys())[0]); self.map_var.trace("w", self.on_map_change)
        map_cb = ttk.Combobox(self.root, textvariable=self.map_var, values=list(self.maps.keys()), state="readonly")
        self.bg_canvas.create_window(PAD+20, LEFT_Y+155, window=map_cb, anchor="nw", width=LEFT_W-40)

        self.map_desc_text = tk.Text(self.root, height=5, font=self.font_main, bg="#f1f5f9", fg=TEXT_DIM, relief=tk.FLAT, padx=10, pady=10)
        self.bg_canvas.create_window(PAD+20, LEFT_Y+200, window=self.map_desc_text, anchor="nw", width=LEFT_W-40, height=120)
        self.update_map_desc()

        self.algo_var = tk.StringVar(value=self.algorithms[0])
        algo_cb = ttk.Combobox(self.root, textvariable=self.algo_var, values=self.algorithms, state="readonly")
        self.bg_canvas.create_window(PAD+20, LEFT_Y+375, window=algo_cb, anchor="nw", width=LEFT_W-40)

        self.run_btn = tk.Button(self.root, text="▶ 开始运行实验", font=self.font_bold, bg=ACCENT_COLOR, fg="white", activebackground=ACCENT_GLOW, relief=tk.FLAT, pady=10, command=self.toggle_experiment)
        self.bg_canvas.create_window(PAD+20, LEFT_Y+420, window=self.run_btn, anchor="nw", width=LEFT_W-40)

        self.timestep_label_var = tk.StringVar(value="当前步数: 0 / 2,050,000")
        ts_label = tk.Label(self.root, textvariable=self.timestep_label_var, font=self.font_bold, fg=ACCENT_COLOR, bg="#f1f5f9")
        self.bg_canvas.create_window(PAD+20, LEFT_Y+520, window=ts_label, anchor="nw")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=2050000)
        self.bg_canvas.create_window(PAD+20, LEFT_Y+560, window=self.progress_bar, anchor="nw", width=LEFT_W-40)

        self.log_area = scrolledtext.ScrolledText(self.root, height=10, font=self.font_mono, bg="#f1f5f9", fg="#1e293b", relief=tk.FLAT)
        self.bg_canvas.create_window(PAD+20, LEFT_Y+660, window=self.log_area, anchor="nw", width=LEFT_W-40, height=180)

        # --- 右侧内容 ---
        RIGHT_X = PAD + LEFT_W + PAD; CURVE_H = 400
        self.bg_canvas.add_card_shape(RIGHT_X, LEFT_Y, 1450-RIGHT_X-PAD, CURVE_H)
        self.bg_canvas.add_canvas_text(RIGHT_X+20, LEFT_Y+15, "📈 实验结果曲线 (Win Rate)", self.font_main, ACCENT_COLOR)
        
        self.fig_curve, self.ax_curve = plt.subplots(figsize=(8, 3.5), dpi=100)
        self.fig_curve.patch.set_facecolor("none") # 设置透明背景
        self.ax_curve.set_facecolor("#f1f5f9")
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, master=self.root)
        self.bg_canvas.create_window(RIGHT_X+20, LEFT_Y+50, window=self.canvas_curve.get_tk_widget(), anchor="nw", width=1450-RIGHT_X-PAD-40, height=CURVE_H-70)

        TSNE_Y = LEFT_Y + CURVE_H + PAD
        self.bg_canvas.add_card_shape(RIGHT_X, TSNE_Y, 1450-RIGHT_X-PAD, 950-TSNE_Y-PAD-40)
        self.bg_canvas.add_canvas_text(RIGHT_X+20, TSNE_Y+15, "🔮 消息空间可视化 (t-SNE)", self.font_main, ACCENT_COLOR)
        
        self.fig_tsne, self.ax_tsne = plt.subplots(figsize=(6, 6), dpi=100)
        self.fig_tsne.patch.set_facecolor("none")
        self.ax_tsne.set_facecolor("#f1f5f9")
        self.canvas_tsne = FigureCanvasTkAgg(self.fig_tsne, master=self.root)
        self.bg_canvas.create_window(RIGHT_X+20, TSNE_Y+50, window=self.canvas_tsne.get_tk_widget(), anchor="nw", width=1450-RIGHT_X-PAD-40, height=950-TSNE_Y-PAD-110)

        self.status_bar = tk.Label(self.root, text="系统就绪 | 正在监控 SMAC 平台...", bd=0, anchor=tk.W, bg="#e2e8f0", fg=TEXT_DIM, font=("Microsoft YaHei", 9), padx=20, pady=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_map_change(self, *args):
        self.update_map_desc(); self.load_data(); self.update_plots()

    def update_plots(self):
        self.plot_curve(); self.plot_tsne()

    def update_map_desc(self):
        self.map_desc_text.config(state=tk.NORMAL)
        self.map_desc_text.delete(1.0, tk.END); self.map_desc_text.insert(tk.END, self.maps.get(self.map_var.get(), ""))
        self.map_desc_text.config(state=tk.DISABLED)

    def toggle_experiment(self):
        if not self.is_running: self.start_experiment()
        else: self.stop_experiment()

    def start_experiment(self):
        map_name = self.map_var.get(); algo_name = self.algo_var.get().lower()
        python_exe = "A:/Conda/envs/smac/python.exe"; main_script = os.path.join(os.path.dirname(__file__), "main.py")
        cmd = [python_exe, main_script, "--env-config=sc2", f"--config={algo_name}", "with", f"env_args.map_name={map_name}"]
        self.log_area.delete(1.0, tk.END); self.log_area.insert(tk.END, f"🚀 正在启动实验: {map_name}...\n")
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, cwd=os.path.dirname(main_script))
            self.is_running = True; self.run_btn.config(text="⏹ 停止实验", bg="#ef4444")
            threading.Thread(target=self.read_logs, daemon=True).start(); self.monitor_progress()
        except Exception as e: messagebox.showerror("启动失败", f"无法启动实验: {str(e)}")

    def stop_experiment(self):
        if self.process:
            self.process.terminate(); self.log_area.insert(tk.END, "\n--- 实验已手动停止 ---\n")
            self.is_running = False; self.run_btn.config(text="▶ 开始运行实验", bg=ACCENT_COLOR)

    def read_logs(self):
        while self.is_running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if line: self.root.after(0, self.update_log_area, line)
        if self.process: self.root.after(0, self.finish_experiment)

    def update_log_area(self, line):
        self.log_area.insert(tk.END, line); self.log_area.see(tk.END)

    def finish_experiment(self):
        self.is_running = False; self.run_btn.config(text="▶ 开始运行实验", bg=ACCENT_COLOR)
        self.load_data(); self.update_plots()

    def monitor_progress(self):
        if not self.is_running: return
        self.load_data(); self.timestep_label_var.set(f"当前步数: {self.current_timestep:,} / 2,050,000")
        self.progress_var.set(self.current_timestep)
        if not hasattr(self, 'last_plot_timestep'): self.last_plot_timestep = -1
        if self.current_timestep - self.last_plot_timestep >= 10000:
            self.update_plots(); self.last_plot_timestep = self.current_timestep
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
                            if "battle_won_mean_T" in self.data_info: self.current_timestep = max(self.data_info["battle_won_mean_T"])
            msg_root = os.path.join(results_root, "messages")
            if os.path.exists(msg_root):
                msg_dirs = os.listdir(msg_root)
                if msg_dirs: self.latest_msg_path = max([os.path.join(msg_root, d) for d in msg_dirs], key=os.path.getmtime)
        except Exception as e: print(f"Error loading data: {e}")

    def plot_curve(self):
        self.ax_curve.clear(); self.ax_curve.set_facecolor("#f1f5f9")
        if hasattr(self, 'data_info') and self.data_info and "battle_won_mean" in self.data_info:
            y = self.data_info["battle_won_mean"]; x = self.data_info.get("battle_won_mean_T", list(range(len(y))))
            self.ax_curve.plot(x, y, color=ACCENT_COLOR, linewidth=2.5, label='COCO')
            self.ax_curve.fill_between(x, y, color=ACCENT_COLOR, alpha=0.1)
            self.ax_curve.legend(facecolor="white", edgecolor=SHADOW_COLOR, labelcolor="#1e293b")
            self.ax_curve.set_xlabel("时间步", color="#475569", fontproperties="Microsoft YaHei", weight="bold")
            self.ax_curve.set_ylabel("平均测试胜率", color="#475569", fontproperties="Microsoft YaHei", weight="bold")
            self.ax_curve.grid(True, linestyle='-', alpha=0.1, color="#cbd5e1")
            self.ax_curve.tick_params(colors="#475569", labelsize=9); self.ax_curve.set_xlim(0, 2050000); self.ax_curve.set_ylim(0, 1.1)
            for spine in self.ax_curve.spines.values(): spine.set_color("#e2e8f0")
        else: self.ax_curve.text(0.5, 0.5, "等待实验数据...", color="#64748b", ha='center', fontproperties="Microsoft YaHei", weight="bold")
        self.fig_curve.tight_layout(); self.canvas_curve.draw()

    def plot_tsne(self):
        self.ax_tsne.clear(); self.ax_tsne.set_facecolor("#f1f5f9")
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
            steps, n_agents, _, msg_dim = msgs.shape; data = msgs.mean(dim=2).permute(1, 0, 2).numpy()
            n_agents, max_step, msg_dim = data.shape; reshaped_data = np.reshape(data.transpose(1, 0, 2), (n_agents * max_step, msg_dim))
            try:
                tsne = TSNE(n_components=2, early_exaggeration=12, perplexity=min(5, reshaped_data.shape[0]-1), init='pca', random_state=42)
                embedded = tsne.fit_transform(reshaped_data); colormaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
                for i in range(n_agents):
                    cmap = get_cmap(colormaps[i % len(colormaps)])
                    for t in range(max_step):
                        self.ax_tsne.scatter(embedded[i*max_step+t, 0], embedded[i*max_step+t, 1], color=cmap(0.4 + 0.6*t/max_step), s=70, edgecolors='white', linewidth=0.3)
            except: pass
        else: self.ax_tsne.text(0.5, 0.5, "等待消息空间数据...", color="#64748b", ha='center', fontproperties="Microsoft YaHei", weight="bold")
        self.ax_tsne.set_xticks([]); self.ax_tsne.set_yticks([])
        for spine in self.ax_tsne.spines.values(): spine.set_visible(False)
        self.fig_tsne.tight_layout(); self.canvas_tsne.draw()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(); style.theme_use('clam')
    style.configure("TCombobox", fieldbackground="white", background=ACCENT_COLOR, foreground=TEXT_PRIMARY, arrowcolor=TEXT_PRIMARY)
    style.configure("TProgressbar", thickness=10, troughcolor="#f1f5f9", background=ACCENT_COLOR)
    app = AutonomousGamingUI(root)
    root.mainloop()
