# Delloyd Malaysia Project

This repository contains solutions and implementations for the **Delloyd Malaysia Internship/Assignment Project**. The work covers computer vision, machine learning, text similarity, and deep learning tasks, divided into multiple parts (A, B, and Q7).

## 📂 Project Structure

```
delloyd_malaysia_project/
│
├── part_a_q1/  
│   └── q1_code.py          # Part A Question 1 solution
│
├── part_b_q3/  
│   └── q3_code.py          # Part B Question 3 solution
│
├── part_b_q4/  
│   └── q4_code.py          # Part B Question 4 solution
│
├── part_b_q5_q6/  
│   ├── q5_main_code.py     # Main code for Question 5  
│   ├── q5_similarity/      
│   │   └── similarity.py   # Text similarity functions  
│   ├── q6_solution/  
│   │   ├── q6_main_code.py # Main code for Question 6  
│   │   └── conftest.py     # Pytest configuration  
│
├── q7_solution/  
│   └── q7_code.py          # Part B Question 7 solution (YOLO + Tkinter GUI)
│
└── README.md               # Project documentation
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/delloyd_malaysia_project.git
cd delloyd_malaysia_project
```

### Install dependencies

This project requires **Python 3.8+** and the following external libraries:

```bash
pip install pillow opencv-python mediapipe pytest torch torchvision ultralytics
```

✅ Note: Built-in libraries such as `os`, `tkinter`, `random`, `re`, and `string` don’t require installation.

## 🚀 Usage

### Part A - Q1

```bash
python part_a_q1/q1_code.py
```

### Part B - Q3

```bash
python part_b_q3/q3_code.py
```

### Part B - Q4

```bash
python part_b_q4/q4_code.py
```

### Part B - Q5

```bash
python part_b_q5_q6/q5_main_code.py
```

### Part B - Q6 (with Pytest)

```bash
pytest part_b_q5_q6/q6_solution/
```

### Part B - Q7 (YOLO + GUI App)

```bash
python q7_solution/q7_code.py
```

Make sure to update the **YOLO model path** in `q7_code.py` if needed.

## 📌 Features

* ✅ **Text similarity** (Q5)
* ✅ **Automated testing with pytest** (Q6)
* ✅ **YOLO object detection with GUI (Tkinter)** (Q7)
* ✅ **Image processing & computer vision (OpenCV, MediaPipe)**

## 🙏 Acknowledgments

Special thanks to **IIUM Malaysia** and **Lords College** for providing the opportunity to work on this project during the internship/training program.
