
 
**David Arnone**

**Future Healthcare AI Engineer**

**Udacity**

09/03/2024

---

## **Motion Compensated Pulse Rate Estimation** 

## Poject goal: ##

The goal of this project was the development of a an algorythm able to evalueate the heartbeat pulsation of human with ppg captors and eccelerometer and to find a method to eveluate the confidence in the evalutation.
The set of data used is extracted from a study named **TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise** [https://arxiv.org/pdf/1409.5181.pdf] which studied the varation of pulsation of individuals under two differents condition of training.

---

The project is devided in two part.

1. Part 1: Pulse Rate Algorithm Project Overview
2. Part 2. Clinical Application


Notebook can be found under src/

validation of test could be found under:
images/passed.png
                   
---

## Further Considerations

Beyond the accomplishments and insights gained from this project, it's crucial to acknowledge certain limitations that could enhance the algorithm's robustness and accuracy.

### Insufficient Data:

The dataset utilized in this study, while valuable, is limited in size and lacks certain essential demographic information such as age, gender, and race. These factors play a pivotal role in the accuracy of PPG signals, and the absence of this data restricts the algorithm's adaptability across diverse populations.

### Outlier Data:

The presence of outliers in the collected data poses a challenge for algorithm performance. Incorporating interpolation techniques or outlier elimination methods could significantly contribute to obtaining more accurate and reliable results.

### Additional Criteria:

Expanding the dataset to include a broader range of criteria, such as skin color, physical condition, and lifestyle, would further enhance the algorithm's generalizability. PPG sensors are sensitive not only to the manner in which they are worn but also to individual characteristics, emphasizing the need for a more diverse and representative dataset.

### Interpolation Techniques:

Consideration should be given to employing interpolation techniques for handling missing or irregular data points. These methods can aid in smoothing out the dataset and refining the algorithm's predictions.

While the current algorithm provides valuable insights, the incorporation of more data and criteria, coupled with advanced data processing techniques, would undoubtedly elevate its performance and applicability in real-world scenarios.

---


## Conclusion and Insights

This project has served as an intriguing exploration into the realms of data engineering and the intricate interplay between technology and physiology. By delving into the development of an algorithm for evaluating heartbeat pulsation, various challenges and considerations have come to light.

### Data Engineering Complexity:

The project underscored the complexity of data engineering, especially when dealing with physiological signals such as PPG. Challenges, including limited dataset size and the absence of crucial demographic information, underscored the need for a more comprehensive approach to data collection.

### Physiological Understanding:

A profound understanding of the physiological aspects, particularly the impact of arm movements during exercise on PPG signals, played a pivotal role in shaping the algorithm. The incorporation of accelerometer data to address this factor showcased the necessity of grounding algorithms in the underlying principles of the data source.

### Caution in Pattern Identification:

The project serves as a reminder of the importance of exercising caution in identifying patterns within a dataset. Without robust validation mechanisms and consideration of various influencing factors, patterns derived from the data may lead to misleading or inaccurate conclusions.

In essence, this project has provided valuable insights into the intricate dance between data, technology, and physiology. As we navigate the complexities of algorithm development, it becomes apparent that success hinges on a holistic understanding of the underlying processes and meticulous validation practices.

---

**Acknowledgments:**
Acknowledge to Udacity for having setup the environment and framework.

---

**Contact Information:**
David Arnone, Zürich , arnoneda@outlook.com.

---
