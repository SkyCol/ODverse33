# ODverse33
Newer YOLO versions are not always better!     

**ODverse33** is a comprehensive benchmark that includes **33 datasets** spanning **11 diverse domains**. It provides a **multi-domain evaluation** for YOLO models, ranging from **YOLOv5 to YOLOv11**.      

The paper, **"ODVerse33: Is the New YOLO Version Always Better? A Multi-Domain Benchmark from YOLO v5 to v11"**, is now available on [*ArXiv*](http://arxiv.org/abs/2502.14314).      



**April 6, 2025 Update:** Fixed minor errors in the manuscript. Updated the manuscript and its figures using LaTeX to make it easier to read. 


        
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

| Metric               | YOLOv5 | YOLOv6 | YOLOv7 | YOLOv8 | YOLOv9 | YOLOv10 | YOLOv11 |
|----------------------|--------|--------|--------|--------|--------|---------|---------|
| **mAP<sub>50</sub>**         | 0.7846 | 0.7675 | 0.7826 | 0.7812 | 0.7913 | 0.7761  | **0.7927** |
| **mAP<sub>50–95</sub>**      | 0.5862 | 0.5498 | 0.5699 | 0.5829 | 0.5902 | 0.5782  | **0.5931** |
| **mAP<sub>small</sub>**      | 0.3722 | 0.3243 | 0.3612 | 0.3735 | **0.3877** | 0.3609  | 0.3855 |
| **mAP<sub>medium</sub>**     | 0.5290 | 0.4822 | 0.5269 | 0.5256 | 0.5357 | 0.5289  | **0.5374** |
| **mAP<sub>large</sub>**      | 0.6487 | 0.6106 | 0.6463 | 0.6481 | 0.6546 | 0.6480  | **0.6559** |

---



More detailed results for 11 diverse domains and 33 datasets can be found in the paper.





