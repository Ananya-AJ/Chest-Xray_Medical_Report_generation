CMPE258 Term Project
Chest x-ray Report Generation and Chatbot
Team:
Ctrl Alt Del

Members:
Ananya Joshi
Sanjana Kothari
Neetha Sherra
Naga Bathula
Project Goals:
The objective of this project is to leverage the power of large language models for generating textual medical reports from chest x-ray images. By utilizing the multimodal capabilities of large language models, we aim to develop an application that takes a chest x-ray image as input and generates a medical report from the x-ray, in a similar fashion as a physician or medical practitioner would. In addition to that, we also provide a chatbot facility built using Microsoft’s Semantic Kernel that can look at the generated report and answer user questions based on the report as well as answer other general queries about chest-related questions. Therefore, it is a complete package that gives the user the diagnosis and observations from chest x-ray as well as give the user the ability to get specific answers to their questions about the report or general chest-related conditions.

Images
Location: ../Images/

https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz

Reports
Location: ../Reports/

https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

App snapshots:
![image](https://github.com/Ananya-AJ/Chest-Xray_Medical_Report_generation/assets/111623197/ed8679c2-c750-49a3-b02e-632c7b6bc76f)


![image](https://github.com/Ananya-AJ/Chest-Xray_Medical_Report_generation/assets/111623197/567f74c7-2242-45cf-a7b3-6904b52ca403)



![image](https://github.com/Ananya-AJ/Chest-Xray_Medical_Report_generation/assets/111623197/e244752c-6a03-44f3-92de-df6df4f95e1b)


To run the applicaton
Clone repo.
Add chest x-ray images in Images folder and reports in Reports folder at root level.
Add OpenAI api key to config.py in Code folder.
Install requiremnts.txt
To run, navigate into the Code folder and use 'streamlit run frontend.py'

Models
https://drive.google.com/drive/folders/1kqEZm906iXJefqE7o03wycBx2D2jw12w?usp=share_link
