# OF-YOLOSeg
Based on YOLOV10, inspired by [AFMA](https://github.com/ShengtianSang/AFMA), the AFMA module is improved to propose OFMA, and the C2f module is improved to propose C2fPro.There is a certain improvement in the segmentation of small targets.

The code is based on the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation?tab=readme-ov-file) 

## File declaration
mmseg/models/decode_heads/OF_YOLOSeg.py：OF-YOLOSeg network structure file
mmse
mmseg/models/losses/focal_loss.py：Contains the improved loss function
AFMA.py：Improved OFMA module
C2fPro：Improved C2fPro module


## Run the codes
```bash
pip install -r requirements.txt
```
