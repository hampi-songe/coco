import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import numpy as np
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap
from PIL import Image, ImageTk
import os
import torch
import warnings
import subprocess
import threading
import sys
import random
import re
import time
import queue

# --- Matplotlib 字体设置 (解决中文显示问题) ---
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

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
PREFERRED_FONT_FILES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

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
        self.sacred_run_id = None
        self.unique_token = None

        self.tsne_busy = False
        self.tsne_requested_step = None
        self.tsne_cache = None
        self.tsne_cache_step = None
        self.log_queue = queue.Queue()
        self.log_flush_scheduled = False
        self.experiment_start_time = None
        self.chart_font = None
        for font_path in PREFERRED_FONT_FILES:
            if os.path.exists(font_path):
                self.chart_font = FontProperties(fname=font_path)
                break
        
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
            "MMM2": "机枪兵,掠夺者,医疗运输机 vs 机枪兵,掠夺者,医疗运输机。MMM地图更大版本，拥有更多单位。",
            "25m": "25个机枪兵 vs 25个机枪兵。一个大规模同质单位对抗地图，考验大规模作战的微操。",
            "3r_vs_2b2m": "3个蟑螂 vs 2个雷神和2个掠夺者。考验小规模不对称战斗中的风筝与集火。",
            "6b_vs_6m": "6个雷神 vs 6个机枪兵。考验重型单位对抗轻型单位时的散开与输出。"
        }
        self.algorithms = ["COCO", "QMIX", "COLA", "TarMAC", "MAIC", "MASIA", "SMS", "NDQ"]; self.platforms = ["SMAC", "Predator-Prey", "Hallway"]

        self.status_bar_var = tk.StringVar(value="系统就绪 | 正在监控 SMAC 平台...")
        self.map_preview_win = None
        self.map_preview_img = None

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        self.bg_canvas = TechBackground(self.root)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)

        # 2. 布局常量
        PAD = 20
        LEFT_W = 380
        HEADER_H = 80
        MIDDLE_H = 550 # 中间配置和图表区的高度
        LOG_H = 220    # 底部日志区的高度
        FULL_W = 1450 - 2 * PAD

        # --- 顶部标题栏 ---
        self.bg_canvas.add_card_shape(PAD, PAD, FULL_W, HEADER_H, r=15)
        self.bg_canvas.add_canvas_text(1450/2, PAD + HEADER_H/2, "自主博弈操作系统", self.font_title, ACCENT_COLOR, anchor="center")

        # --- 中间区域 (Y = 120) ---
        MIDDLE_Y = PAD + HEADER_H + PAD
        
        # 1. 左侧控制面板
        self.bg_canvas.add_card_shape(PAD, MIDDLE_Y, LEFT_W, MIDDLE_H)
        self.bg_canvas.add_canvas_text(PAD+20, MIDDLE_Y+20, "⚙️ 系统配置", self.font_sec, "#334155")
        self.bg_canvas.add_canvas_text(PAD+20, MIDDLE_Y+60, "博弈平台", self.font_main, "#1e293b")
        self.bg_canvas.add_canvas_text(PAD+20, MIDDLE_Y+130, "实验场景", self.font_main, "#1e293b")
        self.bg_canvas.add_canvas_text(PAD+20, MIDDLE_Y+330, "训练算法", self.font_main, "#1e293b")
        self.bg_canvas.add_canvas_text(PAD+20, MIDDLE_Y+430, "📊 实时监控", self.font_sec, "#334155")

        # 放置左侧交互组件
        self.platform_var = tk.StringVar(value=self.platforms[0])
        platform_cb = ttk.Combobox(self.root, textvariable=self.platform_var, values=self.platforms, state="readonly")
        self.bg_canvas.create_window(PAD+20, MIDDLE_Y+85, window=platform_cb, anchor="nw", width=LEFT_W-40)

        self.map_var = tk.StringVar(value=list(self.maps.keys())[0]); self.map_var.trace("w", self.on_map_change)
        map_cb = ttk.Combobox(self.root, textvariable=self.map_var, values=list(self.maps.keys()), state="readonly")
        self.bg_canvas.create_window(PAD+20, MIDDLE_Y+150, window=map_cb, anchor="nw", width=LEFT_W-40)

        self.map_desc_text = tk.Text(self.root, height=2, font=self.font_main, bg="#f1f5f9", fg=TEXT_DIM, relief=tk.FLAT, padx=10, pady=10)
        self.bg_canvas.create_window(PAD+20, MIDDLE_Y+185, window=self.map_desc_text, anchor="nw", width=LEFT_W-40, height=60)
        self.update_map_desc()

        # 新增：地图预览图展示区域
        self.map_preview_label = tk.Label(self.root, bg="#f1f5f9", relief=tk.FLAT)
        self.bg_canvas.create_window(PAD+20, MIDDLE_Y+255, window=self.map_preview_label, anchor="nw", width=LEFT_W-40, height=110)

        self.algo_var = tk.StringVar(value=self.algorithms[0])
        algo_cb = ttk.Combobox(self.root, textvariable=self.algo_var, values=self.algorithms, state="readonly")
        self.bg_canvas.create_window(PAD+20, MIDDLE_Y+405, window=algo_cb, anchor="nw", width=LEFT_W-40)

        self.run_btn = tk.Button(self.root, text="▶ 开始运行实验", font=self.font_bold, bg=ACCENT_COLOR, fg="white", activebackground=ACCENT_GLOW, relief=tk.FLAT, pady=10, command=self.toggle_experiment)
        self.bg_canvas.create_window(PAD+20, MIDDLE_Y+475, window=self.run_btn, anchor="nw", width=LEFT_W-40)

        self.timestep_label_var = tk.StringVar(value="当前步数: 0 / 2,050,000")
        ts_label = tk.Label(self.root, textvariable=self.timestep_label_var, font=self.font_bold, fg=ACCENT_COLOR, bg="#f1f5f9")
        self.bg_canvas.create_window(PAD+20, MIDDLE_Y+520, window=ts_label, anchor="nw")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=2050000)
        self.bg_canvas.create_window(PAD+20, MIDDLE_Y+540, window=self.progress_bar, anchor="nw", width=LEFT_W-40)

        # 2. 右侧图表区 (双正方形并排)
        RIGHT_X = PAD + LEFT_W + PAD
        SQUARE_SIZE = 495 # 保持图表正方形
        
        # 胜率曲线
        self.bg_canvas.add_card_shape(RIGHT_X, MIDDLE_Y, SQUARE_SIZE, SQUARE_SIZE)
        self.bg_canvas.add_canvas_text(RIGHT_X+20, MIDDLE_Y+15, "📈 实验结果曲线 (Win Rate)", self.font_main, ACCENT_COLOR)
        
        # 使用 Figure 替代 plt.subplots
        self.fig_curve = Figure(figsize=(5, 5), dpi=100)
        self.fig_curve.patch.set_facecolor("#ffffff") # 先用白色，确保可见
        self.ax_curve = self.fig_curve.add_subplot(111)
        self.ax_curve.set_facecolor("#f8fafc")
        
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, master=self.bg_canvas) # master 设为 canvas
        self.curve_widget = self.canvas_curve.get_tk_widget()
        self.curve_widget.config(bg="white") # 确保 widget 背景不透明
        self.bg_canvas.create_window(RIGHT_X+20, MIDDLE_Y+50, window=self.curve_widget, anchor="nw", width=SQUARE_SIZE-40, height=SQUARE_SIZE-70)

        # t-SNE 可视化
        TSNE_X = RIGHT_X + SQUARE_SIZE + PAD
        self.bg_canvas.add_card_shape(TSNE_X, MIDDLE_Y, SQUARE_SIZE, SQUARE_SIZE)
        
        # t-SNE 模式选择条 (消息 vs 共识)
        self.tsne_mode_var = tk.StringVar(value="消息")
        self.tsne_mode_frame = tk.Frame(self.root, bg="#f1f5f9", padx=2, pady=2) # 浅灰色背景
        self.bg_canvas.create_window(TSNE_X+20, MIDDLE_Y+10, window=self.tsne_mode_frame, anchor="nw", width=SQUARE_SIZE-40)
        
        rb_style = {
            "font": self.font_bold, 
            "bg": "#f1f5f9", 
            "fg": TEXT_DIM, 
            "selectcolor": "white", 
            "activebackground": "#e2e8f0", 
            "indicatoron": 0, # 改为按钮样式
            "relief": tk.FLAT,
            "padx": 15,
            "pady": 5,
            "width": 18
        }
        
        # 消息可视化按钮
        self.rb_msg = tk.Radiobutton(self.tsne_mode_frame, text="🔮 消息空间", variable=self.tsne_mode_var, value="消息", 
                                    command=self.on_tsne_mode_change, **rb_style)
        self.rb_msg.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # 共识可视化按钮
        self.rb_con = tk.Radiobutton(self.tsne_mode_frame, text="🤝 共识空间", variable=self.tsne_mode_var, value="共识", 
                                    command=self.on_tsne_mode_change, **rb_style)
        self.rb_con.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        self.fig_tsne = Figure(figsize=(5, 5), dpi=100)
        self.fig_tsne.patch.set_facecolor("#ffffff")
        self.ax_tsne = self.fig_tsne.add_subplot(111)
        self.ax_tsne.set_facecolor("#f8fafc")
        
        self.canvas_tsne = FigureCanvasTkAgg(self.fig_tsne, master=self.bg_canvas)
        self.tsne_widget = self.canvas_tsne.get_tk_widget()
        self.tsne_widget.config(bg="white")
        self.bg_canvas.create_window(TSNE_X+20, MIDDLE_Y+50, window=self.tsne_widget, anchor="nw", width=SQUARE_SIZE-40, height=SQUARE_SIZE-70)

        # 初始化样式 (必须在 ax_tsne 初始化后调用)
        self.on_tsne_mode_change()

        # --- 底部日志区域 (Y = 120 + 550 + 20 = 690) ---
        LOG_Y = MIDDLE_Y + MIDDLE_H + PAD
        self.bg_canvas.add_card_shape(PAD, LOG_Y, FULL_W, LOG_H)
        self.bg_canvas.add_canvas_text(PAD+20, LOG_Y+15, "📜 控制台日志输出", self.font_sec, "#334155")
        
        self.log_area = scrolledtext.ScrolledText(self.root, height=8, font=self.font_mono, bg="#f1f5f9", fg="#1e293b", relief=tk.FLAT)
        self.bg_canvas.create_window(PAD+20, LOG_Y+50, window=self.log_area, anchor="nw", width=FULL_W-40, height=LOG_H-70)

        # 底部状态栏
        self.status_bar = tk.Label(self.root, text="系统就绪 | 正在监控 SMAC 平台...", bd=0, anchor=tk.W, bg="#e2e8f0", fg=TEXT_DIM, font=("Microsoft YaHei", 9), padx=20, pady=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化时触发一次预览
        self.root.after(500, self.update_map_preview)

    def on_map_change(self, *args):
        self.update_map_desc()
        self.load_data()
        self.update_plots()
        self.update_map_preview()

    def update_map_preview(self):
        map_name = self.map_var.get()
        img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_pictures")
        img_path = os.path.join(img_dir, f"{map_name}.png")
        
        if not os.path.exists(img_path):
            self.map_preview_label.config(image="", text="暂无地图预览图")
            return

        try:
            # 加载并缩放图片以适应左侧面板宽度
            img = Image.open(img_path)
            # 缩放以适应 340x140 的展示区域 (兼容旧版 Pillow)
            resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
            img.thumbnail((340, 140), resample_filter)
            self.map_preview_img = ImageTk.PhotoImage(img)
            self.map_preview_label.config(image=self.map_preview_img, text="")
            
        except Exception as e:
            print(f"--- 加载地图图片失败: {e} ---")
            self.map_preview_label.config(image="", text="图片加载失败")

    def on_tsne_mode_change(self):
        """当可视化模式改变时更新按钮样式并重绘图表"""
        mode = self.tsne_mode_var.get()
        if mode == "消息":
            self.rb_msg.config(bg="white", fg=ACCENT_COLOR)
            self.rb_con.config(bg="#f1f5f9", fg=TEXT_DIM)
        else:
            self.rb_msg.config(bg="#f1f5f9", fg=TEXT_DIM)
            self.rb_con.config(bg="white", fg=ACCENT_COLOR)
        
        self.plot_tsne()

    def update_plots(self):
        self.plot_curve()
        self.plot_tsne()

    def _normalize_timestep(self, t):
        return 0 if t <= 100 else t

    def _normalize_timeline(self, steps):
        return [self._normalize_timestep(int(v)) for v in steps]

    def request_tsne_update(self, target_step):
        if self.tsne_busy:
            self.tsne_requested_step = target_step
            return

        # 优化：不要在 UI 线程调用耗时的 load_data()
        msg_dir = getattr(self, "latest_msg_path", None)
        token = getattr(self, "unique_token", None)

        self.tsne_busy = True
        self.tsne_requested_step = None
        self.plot_tsne()
        # 将 load_data 的逻辑移入后台线程
        thread = threading.Thread(target=self._compute_tsne_cache_with_load, args=(token, target_step), daemon=True)
        thread.start()

    def _compute_tsne_cache_with_load(self, token, target_step):
        try:
            # 在后台线程刷新路径
            self.load_data()
            msg_dir = getattr(self, "latest_msg_path", None)
            if not msg_dir:
                self.root.after(0, self._finish_tsne_update, None, None)
                return
            
            self._compute_tsne_cache(msg_dir, token, target_step)
        except Exception as e:
            print(f"--- 后台 t-SNE 预加载失败: {e} ---")
            self.root.after(0, self._finish_tsne_update, None, None)

    def _compute_tsne_cache(self, msg_dir, token, target_step):
        try:
            best_path = None
            best_ts = None
            deadline = time.time() + 6.0
            while time.time() < deadline:
                try:
                    msg_files = [f for f in os.listdir(msg_dir) if f.endswith(".pt")]
                except Exception:
                    msg_files = []
                if msg_files:
                    ts_files = sorted([int(f.replace(".pt", "")) for f in msg_files if f.replace(".pt", "").isdigit()])
                    if ts_files:
                        target = int(target_step)
                        tolerance = 120 if target <= 200 else max(300, min(2000, int(target * 0.08)))
                        candidates = [ts for ts in ts_files if abs(ts - target) <= tolerance]
                        if candidates:
                            best_ts = min(candidates, key=lambda ts: abs(ts - target))
                            best_path = os.path.join(msg_dir, f"{best_ts}.pt")
                            if os.path.exists(best_path):
                                break
                time.sleep(0.2)

            if not best_path or not os.path.exists(best_path):
                self.root.after(0, self._finish_tsne_update, None, None)
                return

            data_loaded = torch.load(best_path, map_location="cpu")
            
            # 处理新旧格式数据
            if isinstance(data_loaded, dict):
                msgs_tensor = data_loaded.get("messages")
                consensus_tensor = data_loaded.get("consensus")
            else:
                msgs_tensor = data_loaded
                consensus_tensor = None

            payload = {"best_ts": int(best_ts)}

            # 1. 计算消息 t-SNE
            if msgs_tensor is not None and hasattr(msgs_tensor, "shape") and len(msgs_tensor.shape) == 4:
                steps, n_agents, _, msg_dim = msgs_tensor.shape
                data = msgs_tensor.mean(dim=2).permute(1, 0, 2).numpy()
                n_agents, max_step, msg_dim = data.shape
                reshaped_data = np.reshape(data.transpose(1, 0, 2), (n_agents * max_step, msg_dim))

                max_points = 1600
                if reshaped_data.shape[0] > max_points:
                    idx = np.linspace(0, reshaped_data.shape[0] - 1, num=max_points).astype(int)
                    reshaped_data_use = reshaped_data[idx]
                    t_idx = idx // n_agents
                    agent_idx = idx % n_agents
                else:
                    reshaped_data_use = reshaped_data
                    t_idx = np.arange(reshaped_data.shape[0]) // n_agents
                    agent_idx = np.arange(reshaped_data.shape[0]) % n_agents

                perp = min(30, max(5, (reshaped_data_use.shape[0] - 1) // 5))
                tsne = TSNE(n_components=2, early_exaggeration=12, perplexity=min(perp, reshaped_data_use.shape[0] - 1), init="pca", random_state=42)
                embedded = tsne.fit_transform(reshaped_data_use)
                payload["msg"] = {
                    "embedded": embedded,
                    "agent_idx": agent_idx,
                    "t_idx": t_idx,
                    "n_agents": int(n_agents),
                    "max_step": int(max_step)
                }

            # 2. 计算共识 t-SNE
            if consensus_tensor is not None and hasattr(consensus_tensor, "shape") and len(consensus_tensor.shape) == 3:
                # consensus_tensor shape: [steps, n_agents, consensus_dim=4]
                data = consensus_tensor.permute(1, 0, 2).numpy() # [n_agents, steps, dim]
                n_agents, max_step, con_dim = data.shape
                reshaped_data = np.reshape(data.transpose(1, 0, 2), (n_agents * max_step, con_dim))

                max_points = 1600
                if reshaped_data.shape[0] > max_points:
                    idx = np.linspace(0, reshaped_data.shape[0] - 1, num=max_points).astype(int)
                    reshaped_data_use = reshaped_data[idx]
                    t_idx = idx // n_agents
                    agent_idx = idx % n_agents
                else:
                    reshaped_data_use = reshaped_data
                    t_idx = np.arange(reshaped_data.shape[0]) // n_agents
                    agent_idx = np.arange(reshaped_data.shape[0]) % n_agents

                perp = min(30, max(5, (reshaped_data_use.shape[0] - 1) // 5))
                tsne = TSNE(n_components=2, early_exaggeration=12, perplexity=min(perp, reshaped_data_use.shape[0] - 1), init="pca", random_state=42)
                embedded = tsne.fit_transform(reshaped_data_use)
                payload["con"] = {
                    "embedded": embedded,
                    "agent_idx": agent_idx,
                    "t_idx": t_idx,
                    "n_agents": int(n_agents),
                    "max_step": int(max_step)
                }

            self.root.after(0, self._finish_tsne_update, payload, target_step)
        except Exception:
            self.root.after(0, self._finish_tsne_update, None, None)

    def _finish_tsne_update(self, payload, target_step):
        self.tsne_busy = False
        if payload:
            self.tsne_cache = payload
            self.tsne_cache_step = payload.get("best_ts")
            self.plot_tsne()
        pending = self.tsne_requested_step
        self.tsne_requested_step = None
        if pending is not None:
            self.request_tsne_update(pending)

    def update_map_desc(self):
        self.map_desc_text.config(state=tk.NORMAL)
        self.map_desc_text.delete(1.0, tk.END); self.map_desc_text.insert(tk.END, self.maps.get(self.map_var.get(), ""))
        self.map_desc_text.config(state=tk.DISABLED)

    def toggle_experiment(self):
        if not self.is_running: self.start_experiment()
        else: self.stop_experiment()

    def start_experiment(self):
        # 1. 启动前强制清空所有旧数据和状态
        self.sacred_run_id = None
        self.unique_token = None
        self.data_info = {"test_battle_won_mean": [], "test_battle_won_mean_T": []} # 初始化空数据
        self.latest_msg_path = None
        self.current_timestep = 0
        self.last_plot_timestep = 0
        self.early_fired = set()
        self.last_periodic_bucket = 0
        self.tsne_busy = False
        self.tsne_requested_step = None
        self.tsne_cache = None
        self.tsne_cache_step = None
        self.log_queue = queue.Queue()
        self.log_flush_scheduled = False
        self.experiment_start_time = time.time()
        
        # 2. 立即重置图表显示为“等待中”
        self.update_plots()
        self.timestep_label_var.set("当前步数: 0 / 2,050,000")
        self.progress_var.set(0)
        self.log_area.delete(1.0, tk.END)

        map_name = self.map_var.get(); algo_name = self.algo_var.get().lower()
        python_exe = "A:/Conda/envs/smac/python.exe"; main_script = os.path.join(os.path.dirname(__file__), "main.py")
        # 使用 -u 开启无缓冲模式，确保日志实时传送到 UI
        cmd = [python_exe, "-u", main_script, "--env-config=sc2", f"--config={algo_name}", "with", f"env_args.map_name={map_name}"]
        self.log_area.delete(1.0, tk.END); self.log_area.insert(tk.END, f"🚀 正在启动实验: {map_name}...\n")
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, cwd=os.path.dirname(main_script))
            self.is_running = True; self.run_btn.config(text="⏹ 停止实验", bg="#ef4444")
            threading.Thread(target=self.read_logs, daemon=True).start()
            self.process_log_queue()
            self.monitor_progress()
        except Exception as e: messagebox.showerror("启动失败", f"无法启动实验: {str(e)}")

    def stop_experiment(self):
        if self.process:
            self.process.terminate(); self.log_area.insert(tk.END, "\n--- 实验已手动停止 ---\n")
            self.is_running = False; self.run_btn.config(text="▶ 开始运行实验", bg=ACCENT_COLOR)

    def read_logs(self):
        while self.is_running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if line:
                self.log_queue.put(("line", line))
        if self.process:
            self.log_queue.put(("finished", None))

    def process_log_queue(self):
        appended_lines = []
        curve_dirty = False
        tsne_step = None
        should_finish = False
        processed = 0

        while processed < 200:
            try:
                event_type, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break

            processed += 1
            if event_type == "line":
                appended_lines.append(payload)
                update_curve, update_tsne_step = self.update_log_area(payload, append_to_log=False)
                curve_dirty = curve_dirty or update_curve
                if update_tsne_step is not None:
                    tsne_step = update_tsne_step
            elif event_type == "finished":
                should_finish = True

        if appended_lines:
            self.log_area.insert(tk.END, "".join(appended_lines))
            self.log_area.see(tk.END)

        if curve_dirty:
            self.plot_curve()
        if tsne_step is not None:
            self.request_tsne_update(tsne_step)

        if should_finish:
            self.finish_experiment()
            return

        if self.is_running or not self.log_queue.empty():
            self.root.after(80, self.process_log_queue)

    def update_log_area(self, line, append_to_log=True):
        if append_to_log:
            self.log_area.insert(tk.END, line)
            self.log_area.see(tk.END)
        curve_dirty = False
        tsne_step = None
        
        # 优化：快速预判。如果行中没有关键词，跳过昂贵的正则解析
        if not any(k in line for k in ["Token", "token", "Run ID", "test_battle", "t_env"]):
            return curve_dirty, tsne_step

        # 1. 实时解析 Unique Token (用于定位消息目录)
        token_match = re.search(r"(?:'unique_token':\s*'([^']+)'|Unique Token:\s*([^\s]+))", line)
        if token_match:
            self.unique_token = token_match.group(1) or token_match.group(2)
            print(f"--- 捕捉到 Token: {self.unique_token} ---")
            self.load_data() # 立即定位新目录
        
        # 2. 实时解析 Run ID
        run_id_match = re.search(r'(?:Run ID:|ID\s*")\s*(\d+)', line)
        if run_id_match:
            self.sacred_run_id = run_id_match.group(1)
            print(f"--- 捕捉到 Run ID: {self.sacred_run_id} ---")
        
        # 3. 实时解析胜率 (关键！解决停止才更新的问题)
        win_match = re.search(r"test_battle_won_mean:\s*([\d.]+)", line)
        if win_match:
            win_val = float(win_match.group(1))
            if "test_battle_won_mean" not in self.data_info:
                self.data_info["test_battle_won_mean"] = []
                self.data_info["test_battle_won_mean_T"] = []
            
            # 改进：如果这是第一个点，或者步数较小，强制设为 0
            # 解决“还没训练就标在 10000 步”的问题
            plot_step = self.current_timestep
            if len(self.data_info["test_battle_won_mean"]) == 0 or self.current_timestep < 15000:
                plot_step = 0
            
            # 避免重复点
            last_ts = self.data_info["test_battle_won_mean_T"][-1] if self.data_info["test_battle_won_mean_T"] else None
            if last_ts != plot_step:
                self.data_info["test_battle_won_mean"].append(win_val)
                self.data_info["test_battle_won_mean_T"].append(plot_step)
            print(f"--- 实时抓取胜率: {win_val} (Step: {self.current_timestep}, Plot: {plot_step}) ---")
            
            curve_dirty = True
            tsne_step = self.current_timestep # 测试结束，触发 t-SNE

        # 4. 实时解析时间步
        ts_match = re.search(r"t_env:\s*(\d+)", line)
        if ts_match:
            new_ts = int(ts_match.group(1))
            self.current_timestep = new_ts
            self.timestep_label_var.set(f"当前步数: {self.current_timestep:,} / 2,050,000")
            self.progress_var.set(self.current_timestep)
            
            if not hasattr(self, "early_fired"):
                self.early_fired = set()
            if not hasattr(self, "last_periodic_bucket"):
                self.last_periodic_bucket = 0

            early_thresholds = [50, 200, 500, 1000]
            for t in early_thresholds:
                if self.current_timestep >= t and t not in self.early_fired:
                    print(f"--- 到达早期阈值 {t}，触发同步加载 ---")
                    self.early_fired.add(t)
                    self.load_data()
                    curve_dirty = True
                    if t == 50:
                        tsne_step = 50
                    break

            period = 10000
            bucket = self.current_timestep // period
            if self.current_timestep >= period and bucket > self.last_periodic_bucket:
                print(f"--- 到达 {bucket*period} 步阈值，触发同步加载 ---")
                self.last_periodic_bucket = bucket
                self.load_data()
                curve_dirty = True
                tsne_step = bucket * period

        return curve_dirty, tsne_step

    def finish_experiment(self):
        self.is_running = False; self.run_btn.config(text="▶ 开始运行实验", bg=ACCENT_COLOR)
        self.load_data()
        self.plot_curve()
        self.request_tsne_update(self.current_timestep)

    def monitor_progress(self):
        if not self.is_running: return
        
        # 1. 尝试加载最新数据
        self.load_data()
        
        # 2. 更新步数标签和进度条
        self.timestep_label_var.set(f"当前步数: {self.current_timestep:,} / 2,050,000")
        self.progress_var.set(self.current_timestep)
        
        if not hasattr(self, "early_fired"):
            self.early_fired = set()
        if not hasattr(self, "last_periodic_bucket"):
            self.last_periodic_bucket = 0

        early_thresholds = [50, 200, 500, 1000]
        for t in early_thresholds:
            if self.current_timestep >= t and t not in self.early_fired:
                print(f"检测到早期阈值 {t}，正在更新图表...")
                self.early_fired.add(t)
                self.plot_curve()
                if t == 50:
                    self.request_tsne_update(50)
                break

        period = 10000
        bucket = self.current_timestep // period
        if self.current_timestep >= period and bucket > self.last_periodic_bucket:
            print(f"检测到 {bucket*period} 步阈值，正在更新图表...")
            self.last_periodic_bucket = bucket
            self.plot_curve()
            self.request_tsne_update(bucket * period)
        
        # 4. 每 15 秒轮询一次，降低 IO 压力
        self.root.after(15000, self.monitor_progress)

    def _get_metric_val(self, val):
        """处理 numpy 序列化后的字典格式数据"""
        if isinstance(val, dict) and "value" in val:
            return val["value"]
        return val

    def load_data(self):
        try:
            results_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
            
            # --- 优化 Sacred 数据加载逻辑 ---
            sacred_dir = os.path.join(results_root, "sacred")
            if os.path.exists(sacred_dir):
                if self.sacred_run_id:
                    runs = [self.sacred_run_id]
                else:
                    runs = sorted([d for d in os.listdir(sacred_dir) if d.isdigit()], key=int, reverse=True)
                
                valid_info_found = False
                for run_id in runs:
                    info_path = os.path.join(sacred_dir, run_id, "info.json")
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, "r") as f:
                                data = json.load(f)
                                # 检查多种可能的胜率键名
                                possible_keys = ["test_battle_won_mean", "battle_won_mean", "test_return_mean", "return_mean"]
                                loaded_length = 0
                                current_length = len(self.data_info.get("test_battle_won_mean", [])) if isinstance(getattr(self, "data_info", None), dict) else 0
                                for key in possible_keys:
                                    if key in data and len(data[key]) > 0:
                                        loaded_length = len(data[key])
                                        if loaded_length >= current_length or not self.is_running:
                                            self.data_info = data
                                        valid_info_found = True
                                        print(f"--- 成功加载数据 (Run ID: {run_id}, Key: {key}) ---")
                                        break
                                if valid_info_found: break
                        except Exception as e: 
                            print(f"解析 {info_path} 失败: {e}")
                            continue 
                
                if not valid_info_found:
                    print(f"--- 在最新的 {len(runs)} 个运行中未找到有效的 info.json 数据 ---")

            # --- 优化消息数据加载逻辑 ---
            # 检查两个可能的路径：项目根目录/results 和 src/results
            msg_roots = [
                os.path.join(results_root, "messages"),
                os.path.join(os.path.dirname(results_root), "src", "results", "messages")
            ]
            
            # 如果解析到了 unique_token，则精准定位该文件夹
            found_msg_path = None
            if hasattr(self, 'unique_token') and self.unique_token:
                for msg_root in msg_roots:
                    potential_path = os.path.join(msg_root, self.unique_token)
                    if os.path.exists(potential_path):
                        found_msg_path = potential_path; break
            
            # 运行中如果 token 尚未解析到，则根据启动时间兜底匹配本次实验新建的消息目录
            if not found_msg_path and self.is_running and self.experiment_start_time is not None:
                for msg_root in msg_roots:
                    if os.path.exists(msg_root):
                        msg_dirs = [os.path.join(msg_root, d) for d in os.listdir(msg_root) if os.path.isdir(os.path.join(msg_root, d))]
                        fresh_dirs = [d for d in msg_dirs if os.path.getmtime(d) >= self.experiment_start_time - 5]
                        if fresh_dirs:
                            current_latest = max(fresh_dirs, key=os.path.getmtime)
                            if found_msg_path is None or os.path.getmtime(current_latest) > os.path.getmtime(found_msg_path):
                                found_msg_path = current_latest

            # 非运行状态下允许回退到最新目录做预览
            if not found_msg_path and not self.is_running:
                for msg_root in msg_roots:
                    if os.path.exists(msg_root):
                        msg_dirs = [os.path.join(msg_root, d) for d in os.listdir(msg_root) if os.path.isdir(os.path.join(msg_root, d))]
                        if msg_dirs:
                            current_latest = max(msg_dirs, key=os.path.getmtime)
                            if found_msg_path is None or os.path.getmtime(current_latest) > os.path.getmtime(found_msg_path):
                                found_msg_path = current_latest
            
            self.latest_msg_path = found_msg_path
            if self.latest_msg_path:
                print(f"--- 确认消息路径: {self.latest_msg_path} ---")
            else:
                self.latest_msg_path = None
        except Exception as e:
            print(f"数据加载总异常: {e}")

    def plot_curve(self):
        self.ax_curve.clear()
        self.ax_curve.set_facecolor("#f8fafc")
        
        plot_done = False
        if hasattr(self, 'data_info') and self.data_info:
            for key in ["test_battle_won_mean", "battle_won_mean"]:
                if key in self.data_info and len(self.data_info[key]) > 0:
                    y_raw = self.data_info[key]
                    y = [self._get_metric_val(v) for v in y_raw]
                    x = self._normalize_timeline(self.data_info.get(f"{key}_T", list(range(len(y)))))
                    print(f"--- 正在绘制曲线: {len(y)} 个数据点 ---")
                    self.ax_curve.plot(x, y, color=ACCENT_COLOR, linewidth=2.5, label='COCO', marker='o', markersize=4)
                    self.ax_curve.fill_between(x, y, color=ACCENT_COLOR, alpha=0.1)
                    leg = self.ax_curve.legend(facecolor="white", edgecolor=SHADOW_COLOR)
                    for text in leg.get_texts():
                        text.set_color("#1e293b")
                    plot_done = True
                    break
            
        if plot_done:
            # 明确设置标签字体
            if self.chart_font:
                self.ax_curve.set_xlabel("时间步", color="#475569", fontproperties=self.chart_font, weight="bold")
                self.ax_curve.set_ylabel("平均测试胜率", color="#475569", fontproperties=self.chart_font, weight="bold")
            else:
                self.ax_curve.set_xlabel("时间步", color="#475569", weight="bold")
                self.ax_curve.set_ylabel("平均测试胜率", color="#475569", weight="bold")
            
            self.ax_curve.grid(True, linestyle='-', alpha=0.1, color="#cbd5e1")
            self.ax_curve.tick_params(colors="#475569", labelsize=9)
            
            current_max_x = max(x) if x else 0
            self.ax_curve.set_xlim(0, max(current_max_x * 1.2, 100000, min(self.max_timesteps, current_max_x + 50000)))
            self.ax_curve.set_ylim(-0.05, 1.1)
            for spine in self.ax_curve.spines.values(): spine.set_color("#e2e8f0")
        else:
            text = "等待实验数据..."
            if self.chart_font:
                self.ax_curve.text(0.5, 0.5, text, color="#64748b", ha='center', fontproperties=self.chart_font, weight="bold", transform=self.ax_curve.transAxes)
            else:
                self.ax_curve.text(0.5, 0.5, text, color="#64748b", ha='center', weight="bold", transform=self.ax_curve.transAxes)
        
        self.fig_curve.tight_layout()
        self.canvas_curve.draw()
        # 强制刷新 Tkinter 窗口
        self.canvas_curve.get_tk_widget().update_idletasks()

    def plot_tsne(self):
        self.ax_tsne.clear()
        self.ax_tsne.set_facecolor("#f8fafc")

        mode = self.tsne_mode_var.get() # "消息" 或 "共识"
        mode_key = "msg" if mode == "消息" else "con"
        
        cache = getattr(self, "tsne_cache", None)
        if cache and mode_key in cache:
            data_payload = cache[mode_key]
            embedded = data_payload["embedded"]
            agent_idx = data_payload["agent_idx"]
            t_idx = data_payload["t_idx"]
            n_agents = int(data_payload["n_agents"])
            max_step = int(data_payload["max_step"])
            
            colormaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrBr', 'PuRd', 'GnBu']
            try:
                from matplotlib import cm
                cm_get = cm.get_cmap
            except Exception:
                cm_get = None

            colors = []
            for a, tt in zip(agent_idx, t_idx):
                cmap_name = colormaps[int(a) % len(colormaps)]
                cmap = cm_get(cmap_name) if cm_get else get_cmap(cmap_name)
                # 颜色深浅随时间步变化
                frac = 0.4 + 0.6 * (float(tt) / max(1, max_step))
                colors.append(cmap(frac))

            self.ax_tsne.scatter(embedded[:, 0], embedded[:, 1], c=colors, s=55, edgecolors='white', linewidth=0.3, alpha=0.8)
            
            title_text = f"Step {cache.get('best_ts', '')} ({mode})"
            if self.chart_font:
                self.ax_tsne.set_title(title_text, color="#475569", fontproperties=self.chart_font, weight="bold", fontsize=10)
            else:
                self.ax_tsne.set_title(title_text, color="#475569", fontproperties="DejaVu Sans", weight="bold", fontsize=10)
        else:
            text = "t-SNE 计算中..." if getattr(self, "tsne_busy", False) else f"等待{mode}空间数据..."
            if self.chart_font:
                self.ax_tsne.text(0.5, 0.5, text, color="#64748b", ha='center', fontproperties=self.chart_font, weight="bold", transform=self.ax_tsne.transAxes)
            else:
                self.ax_tsne.text(0.5, 0.5, text, color="#64748b", ha='center', fontproperties="DejaVu Sans", weight="bold", transform=self.ax_tsne.transAxes)
        
        self.ax_tsne.set_xticks([]); self.ax_tsne.set_yticks([])
        for spine in self.ax_tsne.spines.values(): spine.set_visible(False)
        self.fig_tsne.tight_layout()
        self.canvas_tsne.draw()
        self.canvas_tsne.get_tk_widget().update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(); style.theme_use('clam')
    style.configure("TCombobox", fieldbackground="white", background=ACCENT_COLOR, foreground=TEXT_PRIMARY, arrowcolor=TEXT_PRIMARY)
    style.configure("TProgressbar", thickness=10, troughcolor="#f1f5f9", background=ACCENT_COLOR)
    app = AutonomousGamingUI(root)
    root.mainloop()
