# 👷‍♂️ PPE-Detection-YOLOv8
Detection of Personal Protective Equipment (PPE) using computer vision in industrial environments. This project automates the verification of safety protocols by cross-referencing real-time visual data with regulatory documentation.

[!NOTE] *This project was developed in collaboration with [@samuelgtrz](https://github.com/samuelgtrz) and [@Roquedearcos](https://github.com/Roquedearcos), with mentorship and tracking by [Gradiant](https://www.gradiant.org). This version includes several enhancements and optimizations over the original baseline. We also had the collaboration of the University of Santiago de Compostela (USC).*

🚩 **The Problem: Occupational Risk Management**

Ensuring the correct use of PPE (helmets, vests, gloves and boots) is critical in industrial settings. Traditionally, this verification is performed by supervisors through manual inspections, which are:
* Costly: Requires dedicated personnel time.
* Invasive: Can disrupt workflow and worker privacy.
* Prone to error: Human fatigue can lead to missed safety violations.

🎯 **The Goal**: Automate and optimize the monitoring of PPE compliance to ensure a safer, more efficient workplace.

💡 **Our Solution**

We have approached this challenge by integrating two distinct technologies into a single automated pipeline:

1. Document Intelligence (NLP)
A module dedicated to parsing regulatory PDF documents. Using Natural Language Processing, the system automatically identifies which specific PPE are required for a particular restricted area.

2. Intelligent Verification (Computer Vision)
Once the requirements are set, the system uses a smart camera feed to detect the presence (or absence) of the required equipment on workers entering the area. This is powered by YOLOv8, ensuring high speed and accuracy.


### 📄 NLP Module (Regulation Extraction)
This module acts as the "brain" of the system, interpreting complex safety documents.
* **Powered by:** Google Gemini 2.5 Flash.
* **Function:** Converts unstructured PDF text into actionable JSON data.
* **Features:**
    * Automatic PPE categorization.
    * Mandatory vs. Optional usage detection.
    * Interactive GUI for file selection.

### 🔍 CV Module (PPE Detection & Compliance)
This module acts as the "eyes" of the system, verifying real-time safety compliance through visual analysis.
* **Powered by:** YOLOv8 (Ultralytics) fine-tuned for safety equipment.
* **Function:** Identifies PPE items in images and validates them against extracted regulations.
* **Features:** Real-time Object Detection: High-accuracy identification of helmets, vests, gloves, and boots.
   * Automated Compliance Check: Cross-references detected items with area-specific requirements from the NLP module.
   * Access Control Logic: Automatically generates "Authorized" or "Denied" verdicts based on mandatory equipment.
   * Visual Reporting: Generates annotated images and detailed CSV reports for safety audits.
 
## 📺 Video Demonstration
Due to file size limitations, the full high-resolution demonstration is hosted on OneDrive.

👉 **[Watch the Demo Video Here](https://nubeusc-my.sharepoint.com/personal/breixo_brea_rai_usc_es/_layouts/15/stream.aspx?id=%2Fpersonal%2Fbreixo%5Fbrea%5Frai%5Fusc%5Fes%2FDocuments%2FGRIA%2F4%C2%BA%20CURSO%2E%201%C2%BA%20CUADRIMESTRE%2FProxecto%20Integrador%20II%2Fproxecto%5Ffinal%2Fdemo%2Emp4&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&ga=1&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E5d5b40f2%2De6c7%2D4663%2D9228%2Dc6229afd38c0)**

### What's shown in the video:
* **Regulation Extraction:** Processing a safety PDF into JSON.
* **Area Selection:** Dynamic selection of work zones.
* **PPE Detection:** Real-time analysis of worker images.
* **Compliance Report:** Generation of access granted/denied verdicts.
