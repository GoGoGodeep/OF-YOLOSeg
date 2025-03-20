# 🚀 OF-YOLOSeg

**Base Model**: YOLOv10   
**Framework**: [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)  
**Inspired by**: [AFMA](https://github.com/ShengtianSang/AFMA) with architectural improvements

## 🧠 核心创新
✅ **OFMA 模块**  
`(基于AFMA改进)` 优化特征聚合策略，提升小目标特征保留能力  
✅ **C2fPro 模块**  
`(C2f升级版)` 增强多尺度特征融合效率，降低计算开销  
✅ **小目标专项优化**  
通过双路径注意力机制改进边界预测精度

## 📂 文件架构
```bash
├── mmseg/
│   ├── models/decode_heads/OF_YOLOSeg.py   # 🌐 核心网络架构
│   ├── models/losses/focal_loss.py         # ⚖️ 改进的Focal Loss
├── AFMA.py                                 # 🔥 OFMA模块实现
├── C2fPro.py                               # 💎 C2fPro模块实现
```

## 🛠️ 快速开始
```bash
# 环境安装
pip install -r requirements.txt

# 训练指令（示例）
python tools/train.py configs/of_yoloseg/of_yoloseg_r50.py
```

---

> 📌 注：需配合MMSegmentation基础环境使用
