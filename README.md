## Практическая работа 1
Набор данных представляет собой средняя температура воздуха для каждого дня 2023 года. Были созданы 5 наборов данных для обучения и 2 набора для тестирования. В каждом наборе были добавлены шумы и аномалии, которые были убраны в этапе предварительной обработки. 
<p>Для обучения модели был использован модель случайного леса из библиотеки <i>scikit-learn</i>.</p>

Для коректной работы программы необходимо загрузить следующие библиотеки :

```python
import gradio as gr
import os
import moviepy.editor as mp 
from faster_whisper import WhisperModel
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import sumy
import nltk
```
