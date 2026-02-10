# 👷‍♂️ PPE-Detection-YOLOv8
Detection of Personal Protective Equipment (PPE) using computer vision in industrial environments. This project automates the verification of safety protocols by cross-referencing real-time visual data with regulatory documentation.

[!NOTE] *This project was developed in collaboration with [@samuelgtrz](https://github.com/samuelgtrz) and [@Roquedearcos](https://github.com/Roquedearcos), with mentorship and tracking by [Gradiant](https://www.gradiant.org). This version includes several enhancements and optimizations over the original baseline. We also had the collaboration of the University of Santiago de Compostela (USC).*

🚩 **The Problem: Occupational Risk Management**

Ensuring the correct use of PPE (helmets, vests, gloves and boots) is critical in industrial settings. Traditionally, this verification is performed by supervisors through manual inspections, which are:
- Costly: Requires dedicated personnel time.
- Invasive: Can disrupt workflow and worker privacy.
- Prone to error: Human fatigue can lead to missed safety violations.

🎯 **The Goal**: Automate and optimize the monitoring of PPE compliance to ensure a safer, more efficient workplace.

💡 **Our Solution**

We have approached this challenge by integrating two distinct technologies into a single automated pipeline:

1. Document Intelligence (NLP)
A module dedicated to parsing regulatory PDF documents. Using Natural Language Processing, the system automatically identifies which specific PPE are required for a particular restricted area.

2. Intelligent Verification (Computer Vision)
Once the requirements are set, the system uses a smart camera feed to detect the presence (or absence) of the required equipment on workers entering the area. This is powered by YOLOv8, ensuring high speed and accuracy.

🛠️ **Tech Stack**

Vision: YOLOv8 (Ultralytics)

Language: Python

NLP: Gemini

Tracking: Gradiant Methodology
