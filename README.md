# ODverse33
Newer YOLO versions are not always better!     

**ODverse33** is a comprehensive benchmark that includes **33 datasets** spanning **11 diverse domains**. It provides a **multi-domain evaluation** for YOLO models, ranging from **YOLOv5 to YOLOv11**.      

The paper, **"ODVerse33: Is the New YOLO Version Always Better? A Multi-Domain Benchmark from YOLO v5 to v11"**, is now available on [*ArXiv*](http://arxiv.org/abs/2502.14314).      

*Notice: We are making some updates and the datasets will be released soon.*    

**April 6, 2025 Update:** Fixed minor errors in the manuscript. Updated the manuscript and its figures using LaTeX to make it easier to read. The updated paper is currently in replacement procedure.


        
A Timeline of YOLO series detectors from v1 to v11:    

<p align="left">
  <img src="https://github.com/user-attachments/assets/76c06acb-6029-4402-b858-38d0cca41046" width="100%" height="200%">
</p>

Results on ODverse 33 test sets and COCO validation set:
![Image](https://github.com/user-attachments/assets/ad4dbdee-dcdc-4d71-9d4a-6b26fe9d7878)

### 🌐 Covered Domains

<table>
  <tr>
    <td>Autonomous Driving</td>
    <td>Agricultural</td>
    <td>Underwater</td>
  </tr>
  <tr>
    <td>Medical</td>
    <td>Videogame</td>
    <td>Industrial</td>
  </tr>
  <tr>
    <td>Aerial</td>
    <td>Wildlife</td>
    <td>Retail</td>
  </tr>
  <tr>
    <td>Microscopic</td>
    <td>Security</td>
    <td></td>
  </tr>
</table>





---

### 📊 Overall Performance on ODverse33 Test Sets (mAP)

| Metric | YOLOv5 | YOLOv6 | YOLOv7 | YOLOv8 | YOLOv9 | YOLOv10 | YOLOv11 |
|---------------------|--------|--------|--------|--------|--------|---------|---------|
| **mAP<sub>50</sub>** | 0.78464 | 0.76745 | 0.7826 | 0.78118 | 0.7913 | 0.77607 | **0.79274** |
| **mAP<sub>50–95</sub>** | 0.586202 | 0.549798 | 0.569941 | 0.582927 | 0.590204 | 0.578214 | **0.593098** |
| **mAP<sub>small</sub>** | 0.37218 | 0.324318 | 0.361175 | 0.373528 | **0.387723** | 0.360913 | 0.385532 |
| **mAP<sub>medium</sub>** | 0.529017 | 0.482247 | 0.526936 | 0.525615 | 0.535742 | 0.528904 | **0.537404** |
| **mAP<sub>large</sub>** | 0.648672 | 0.610599 | 0.646268 | 0.648142 | 0.654562 | 0.647964 | **0.655859** |

More detailed results for 11 diverse domains and 33 datasets can be found in the paper.





