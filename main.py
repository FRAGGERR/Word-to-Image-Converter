import base64
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Take a img which have NO BACKGROUND (if the image have bg remove it), copy the image path and past here down.
input_img = "your_img_path"
with open(input_img, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Download the CSV file and past the path of the CSV file which is in your sys.
downloaded_csv_path = "/Users/hardikchhipa/Desktop/NLP/Projects/spooky-author-identification/train.csv"
train = pd.read_csv(downloaded_csv_path)



image_bytes = base64.b64decode(encoded_string)
image = Image.open(io.BytesIO(image_bytes))

text_data = ' '.join(train['text'].values)

wc = WordCloud(mask=np.array(image), stopwords=STOPWORDS, background_color='white')
wc.generate(text_data)

plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
